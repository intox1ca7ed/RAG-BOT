from __future__ import annotations

from typing import List, Tuple

import logging
import re
import numpy as np

from .config import Config, DEFAULT_CONFIG
from .embeddings import EmbeddingModel
from .index_store import IndexStore
from .reranker import Reranker

logger = logging.getLogger(__name__)

# NOTE: Routing rewrite for stability: intent detection (contact/form), tag/country-aware
# doc filtering, hierarchical doc shortlist + chunk scoring, and clearer debug signals to
# avoid keyword traps (e.g., furniture vs contacts) while keeping deterministic behavior.

# Retrieval tuning constants
DOC_SHORTLIST = 8
MIN_FILTERED_CANDIDATES = 10
STRICT_FILTER_CONFIDENCE = 0.8
CONTACT_INTENT_BOOST = 0.25
PROCESSING_TIME_BOOST = 0.4
CONTACT_STRICT_CONFIDENCE = 0.75
CONTACT_DOC_IDS = {"rag_corpus_contacts_txt", "rag_corpus_consulate_contacts_txt"}
PROCESSING_TERMS = ["how long", "processing", "business days", "days", "urgent", "express", "time to make"]
PROCESSING_COUNTRY_TERMS = {"china", "prc", "chinese"}
FURNITURE_TRAP_TAGS = {"furniture", "tour"}
POST_FILTER_MAX_CHUNKS = 3
PROCESSING_CHUNK_KEYWORDS = [
    "business day",
    "days",
    "urgent",
    "express",
    "processing time",
    "processing",
    "term",
]
CONTACT_DOC_ID_KEYWORD = "contact"


def _tags_set(meta_tags: str | list | None) -> set[str]:
    if meta_tags is None:
        return set()
    if isinstance(meta_tags, list):
        items = meta_tags
    else:
        items = str(meta_tags).split(",")
    return {t.strip().lower() for t in items if t and str(t).strip()}


def _merged_tags(meta: dict) -> set[str]:
    tags = _tags_set(meta.get("tags"))
    topic = meta.get("topic")
    section = meta.get("section")
    if topic:
        tags.add(str(topic).lower())
    if section:
        tags.add(str(section).lower())
    return tags


def _infer_countries(meta: dict) -> set[str]:
    tags = _tags_set(meta.get("tags"))
    country_field = str(meta.get("country", "")).lower()
    countries = set()
    if country_field in {"china", "japan", "taiwan"}:
        countries.add(country_field)
    for kw in ("china", "japan", "taiwan"):
        if kw in tags:
            countries.add(kw)
    return countries


def _detect_allowed_tags(question: str) -> tuple[set[str], float]:
    q = question.lower()
    allowed = set()
    confidence = 0.0
    time_terms = any(kw in q for kw in ["how long", "processing", "time", "days"])
    if "china" in q:
        allowed.add("china")
        confidence = max(confidence, 0.9)
    if "japan" in q:
        allowed.add("japan")
        confidence = max(confidence, 0.9)
    if "taiwan" in q:
        allowed.add("taiwan")
        confidence = max(confidence, 0.9)
    if any(k in q for k in ["furniture", "tour", "guangzhou", "foshan"]):
        allowed.add("furniture")
        confidence = max(confidence, 0.7)
    if any(k in q for k in ["legalization", "apostille", "legal"]):
        allowed.add("legalization")
        confidence = max(confidence, 0.7)
    if "visa" in q:
        allowed.add("visa")
        confidence = max(confidence, 0.6)
    if any(k in q for k in ["contact", "office", "address", "phone", "email", "where"]):
        allowed.add("contact")
        confidence = max(confidence, 0.7)
    if time_terms:
        confidence = min(confidence, 0.6)  # avoid over-strict tag gating for timing questions
    return allowed, confidence


def _detect_intent(question: str) -> tuple[str | None, float, str]:
    q = question.lower()
    contact_terms = ["contact", "office", "address", "phone", "email", "where", "find", "location", "wechat"]
    if any(term in q for term in contact_terms):
        return "contact", 0.9, "contact_keywords"
    proc_terms = [t for t in PROCESSING_TERMS if t in q]
    if proc_terms:
        return "processing_time", 0.8, f"processing_terms:{'|'.join(proc_terms)}"
    if "form" in q or "application" in q:
        return "form", 0.6, "form_keyword"
    return None, 0.0, "none"


def _keyword_bonus(question: str, meta: dict) -> float:
    q_tokens = {w for w in question.lower().split() if len(w) > 2}
    title_tokens = {w.strip(",.") for w in str(meta.get("title", "")).lower().split()}
    tag_tokens = _tags_set(meta.get("tags"))
    overlap = (q_tokens & title_tokens) | (q_tokens & tag_tokens)
    return 0.05 * len(overlap)


def _doc_matches_filters(
    allowed_tags: set[str],
    allowed_conf: float,
    countries: set[str],
    tags: set[str],
    prefer_china_default: bool,
) -> bool:
    if prefer_china_default:
        # If we defaulted to China, exclude docs explicitly tagged with other countries.
        if ("japan" in countries or "taiwan" in countries) or ("japan" in tags or "taiwan" in tags):
            return False
        # Allow China or country-unknown docs.
    if not allowed_tags:
        return True
    if allowed_conf >= STRICT_FILTER_CONFIDENCE:
        # strict: require intersection
        if countries and allowed_tags & countries:
            return True
        return bool(allowed_tags & tags)
    # permissive
    if not tags:
        return True
    return bool(allowed_tags & tags) or bool(allowed_tags & countries)


def _score_documents(
    question: str,
    query_vec: np.ndarray,
    index_store: IndexStore,
    allowed_tags: set[str],
    allowed_conf: float,
    intent: str | None,
    prefer_china_default: bool,
    contact_strict: bool,
    contact_doc_ids: set[str],
    contact_allowed_docs: set[str],
    processing_country: str | None,
    exclude_furniture: bool,
    query_has_furniture: bool,
    debug: bool,
    exclusions: list[str],
    include_reasons: list[str],
) -> tuple[dict, dict, list[str]]:
    vectors = index_store.vectors
    chunks = index_store.chunks
    doc_indices: dict[str, list[int]] = {}
    for idx, chunk in enumerate(chunks):
        doc_id = chunk.get("metadata", {}).get("doc_id", f"doc-{idx}")
        doc_indices.setdefault(doc_id, []).append(idx)

    doc_scores: dict[str, float] = {}
    doc_meta_map: dict[str, dict] = {}
    filtered_doc_ids: list[str] = []

    q_lower = question.lower()
    for doc_id, idxs in doc_indices.items():
        meta = chunks[idxs[0]].get("metadata", {})
        tags = _merged_tags(meta)
        countries = _infer_countries(meta)

        if contact_strict:
            if doc_id not in contact_allowed_docs:
                exclusions.append(f"{doc_id}: excluded (contact strict allowlist)")
                continue
            include_reasons.append(f"{doc_id}: included (contact allowlist)")

        if processing_country:
            if not (processing_country in countries or processing_country in tags):
                exclusions.append(f"{doc_id}: excluded (processing intent country mismatch)")
                continue
            if not ({"visa", "processing", "main_page", "general"} & tags):
                exclusions.append(f"{doc_id}: excluded (processing intent missing visa/processing/main_page tag)")
                continue
            other_countries = {"japan", "taiwan"} & (countries | tags)
            if other_countries:
                exclusions.append(f"{doc_id}: excluded (processing intent other country tags: {','.join(sorted(other_countries))})")
                continue

        if exclude_furniture and not query_has_furniture and (FURNITURE_TRAP_TAGS & tags):
            exclusions.append(f"{doc_id}: excluded (furniture/tour trap)")
            continue

        if not contact_strict and not _doc_matches_filters(allowed_tags, allowed_conf, countries, tags, prefer_china_default):
            continue
        filtered_doc_ids.append(doc_id)

        doc_vecs = vectors[idxs]
        sims = doc_vecs @ query_vec.reshape(-1, 1)
        max_sim = float(np.max(sims)) if sims.size else 0.0
        score = max_sim + _keyword_bonus(question, meta)
        if intent == "contact" and ("contact" in tags or "locations" in tags or "hours" in tags):
            score += CONTACT_INTENT_BOOST
        if intent == "form" and ("form" in tags or meta.get("doc_type") == "csv"):
            score += 0.05
        if any(t in question.lower() for t in ["how long", "processing", "business days", "days"]) and "visa" in allowed_tags:
            for idx in idxs:
                text_lower = chunks[idx].get("text", "").lower()
                if "day" in text_lower and "time" in text_lower:
                    score += PROCESSING_TIME_BOOST
                    break
        score = max(-1.0, min(1.0, score))
        doc_scores[doc_id] = score
        doc_meta_map[doc_id] = meta

    return doc_scores, doc_indices, filtered_doc_ids


def retrieve(
    question: str,
    index_store: IndexStore,
    embed_model: EmbeddingModel,
    top_k: int | None = None,
    config: Config | None = None,
    debug: bool = False,
    rerank: bool = False,
    rerank_mode: str = "off",
    rerank_top_n: int | None = None,
    rerank_top_k: int | None = None,
    reranker_model: str | None = None,
) -> tuple[List[tuple[dict, float]], bool, dict]:
    cfg = config or DEFAULT_CONFIG
    k = top_k or cfg.top_k
    rerank_top_n_val = rerank_top_n or 30
    rerank_top_k_val = rerank_top_k or k
    allowed_tags, allowed_conf = _detect_allowed_tags(question)
    intent, intent_conf, intent_rule = _detect_intent(question)
    processing_focus = intent == "processing_time"
    prefer_china_default = False
    if "visa" in allowed_tags and not ({"china", "japan", "taiwan"} & allowed_tags):
        # Default to China priority when no country specified
        allowed_tags.add("china")
        prefer_china_default = True
        allowed_conf = max(allowed_conf, 0.8)
    # Strengthen contact intent routing: force tag gate to contact/locations when confident
    if intent == "contact" and intent_conf >= STRICT_FILTER_CONFIDENCE:
        allowed_tags = {"contact", "locations", "hours"}
        allowed_conf = 1.0
    # Processing-time intent country restriction
    processing_country: str | None = None
    q_lower = question.lower()
    if processing_focus and ("china" in allowed_tags or any(term in q_lower for term in PROCESSING_COUNTRY_TERMS)):
        processing_country = "china"

    contact_strict = intent == "contact" and intent_conf >= CONTACT_STRICT_CONFIDENCE
    rerank_enabled = False
    reranker_model_name = reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device = None
    rerank_changed = 0
    rerank_used = False
    rerank_mode_val = rerank_mode or "off"
    if rerank_mode_val == "on" or (rerank_mode_val == "auto" and embed_model.backend == "sentence-transformers"):
        rerank_enabled = True
    contact_allowed_docs: set[str] = set()
    contact_include_reasons: list[str] = []
    exclusions: list[str] = []
    if contact_strict:
        doc_indices_probe: dict[str, list[int]] = {}
        for idx, chunk in enumerate(index_store.chunks):
            doc_id = chunk.get("metadata", {}).get("doc_id", f"doc-{idx}")
            doc_indices_probe.setdefault(doc_id, []).append(idx)
        for doc_id, idxs in doc_indices_probe.items():
            meta = index_store.chunks[idxs[0]].get("metadata", {})
            tags = _merged_tags(meta)
            doc_id_lower = doc_id.lower()
            if FURNITURE_TRAP_TAGS & tags:
                exclusions.append(f"{doc_id}: excluded (contact strict furniture/tour tag)")
                continue
            if doc_id in CONTACT_DOC_IDS:
                contact_allowed_docs.add(doc_id)
                contact_include_reasons.append(f"{doc_id}: included (priority contact doc_id)")
                continue
            if CONTACT_DOC_ID_KEYWORD in doc_id_lower:
                contact_allowed_docs.add(doc_id)
                contact_include_reasons.append(f"{doc_id}: included (doc_id contains '{CONTACT_DOC_ID_KEYWORD}')")
                continue
            if "contact" in tags:
                contact_allowed_docs.add(doc_id)
                contact_include_reasons.append(f"{doc_id}: included (tag 'contact')")
                continue
            exclusions.append(f"{doc_id}: excluded (contact strict not a contact doc)")

    query_vec = embed_model.encode([question])[0]
    # Defensive normalization in case upstream backend changes
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        norm = 1.0
    query_vec = (query_vec / norm).astype(np.float32)

    exclude_furniture = "visa" in allowed_tags and ({"china", "japan", "taiwan"} & allowed_tags)
    doc_scores, doc_indices, filtered_doc_ids = _score_documents(
        question,
        query_vec,
        index_store,
        allowed_tags,
        allowed_conf,
        intent,
        prefer_china_default,
        contact_strict,
        CONTACT_DOC_IDS,
        contact_allowed_docs,
        processing_country,
        exclude_furniture,
        query_has_furniture="furniture" in q_lower or "tour" in q_lower,
        debug=debug,
        exclusions=exclusions,
        include_reasons=contact_include_reasons,
    )

    filter_warning = False
    fallback_used = False
    fallback_reasons: list[str] = []
    allow_filter_fallback = not contact_strict
    if (
        allow_filter_fallback
        and allowed_tags
        and len(filtered_doc_ids) < MIN_FILTERED_CANDIDATES
        and allowed_conf < STRICT_FILTER_CONFIDENCE
    ):
        filter_warning = True  # not enough docs; we'll fallback to all docs
        fallback_used = True
        fallback_reasons.append("doc filter broadened (insufficient candidates)")
        # Re-run without tag filter but keep intent boosts
        doc_scores, doc_indices, filtered_doc_ids = _score_documents(
            question,
            query_vec,
            index_store,
            set(),
            0.0,
            intent,
            prefer_china_default=False,
            contact_strict=contact_strict,
            contact_doc_ids=CONTACT_DOC_IDS,
            contact_allowed_docs=contact_allowed_docs,
            processing_country=processing_country,
            exclude_furniture=exclude_furniture,
            query_has_furniture="furniture" in q_lower or "tour" in q_lower,
            debug=debug,
            exclusions=exclusions,
            include_reasons=contact_include_reasons,
        )

    if not doc_scores and allow_filter_fallback:
        # final fallback: unfiltered doc list
        fallback_used = True
        fallback_reasons.append("doc filter removed (no candidates)")
        doc_scores, doc_indices, filtered_doc_ids = _score_documents(
            question,
            query_vec,
            index_store,
            set(),
            0.0,
            intent,
            prefer_china_default=False,
            contact_strict=contact_strict,
            contact_doc_ids=CONTACT_DOC_IDS,
            contact_allowed_docs=contact_allowed_docs,
            processing_country=None,
            exclude_furniture=exclude_furniture,
            query_has_furniture="furniture" in q_lower or "tour" in q_lower,
            debug=debug,
            exclusions=exclusions,
            include_reasons=contact_include_reasons,
        )

    shortlisted = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:DOC_SHORTLIST]
    shortlisted_ids = {doc_id for doc_id, _ in shortlisted}
    if processing_focus:
        for doc_id in doc_indices.keys():
            if "main_page" in doc_id or "main-page" in doc_id:
                shortlisted_ids.add(doc_id)

    def _chunk_allowed(meta: dict) -> bool:
        tags = _tags_set(meta.get("tags"))
        countries = _infer_countries(meta)
        doc_id_val = meta.get("doc_id", "")
        if contact_strict:
            if contact_allowed_docs:
                return doc_id_val in contact_allowed_docs
            return ("contact" in tags) or doc_id_val in CONTACT_DOC_IDS
        if processing_country:
            if not (processing_country in tags or processing_country in countries):
                return False
            if {"japan", "taiwan"} & (countries | tags):
                return False
            if not ({"visa", "processing", "main_page", "general"} & tags):
                return False
        if exclude_furniture and not ("furniture" in q_lower or "tour" in q_lower):
            if FURNITURE_TRAP_TAGS & tags:
                return False
        return True

    # Gather chunk scores inside shortlisted docs
    vectors = index_store.vectors
    chunks = index_store.chunks
    chunk_hits: list[tuple[dict, float]] = []
    for doc_id in shortlisted_ids:
        for idx in doc_indices.get(doc_id, []):
            sim = float(vectors[idx] @ query_vec)
            sim = max(-1.0, min(1.0, sim))
            meta = chunks[idx].get("metadata", {})
            tags = _tags_set(meta.get("tags"))
            if intent == "contact" and ("contact" in tags or "locations" in tags or "hours" in tags):
                sim += 0.2
            if intent == "form" and ("form" in tags or meta.get("doc_type") == "csv"):
                sim += 0.05
            if processing_focus:
                text_lower = chunks[idx].get("text", "").lower()
                if "day" in text_lower and "time" in text_lower:
                    sim += 0.3
            sim = max(-1.0, min(1.0, sim))
            chunk_hits.append((chunks[idx], sim))

    chunk_hits.sort(key=lambda x: x[1], reverse=True)
    # Post-filter chunks to enforce country/topic constraints and limit context size
    filtered_hits = [(c, s) for c, s in chunk_hits if _chunk_allowed(c.get("metadata", {}))]

    rerank_table: list[str] = []
    if rerank_enabled and filtered_hits:
        reranker = Reranker.get(reranker_model_name)
        if not reranker.available():
            rerank_enabled = False
            fallback_reasons.append("rerank unavailable (model load failed)")
        else:
            rerank_device = reranker.device
            top_n = min(len(filtered_hits), rerank_top_n_val)
            candidates = filtered_hits[:top_n]
            texts = [c["text"] for c, _ in candidates]
            scores = reranker.score(question, texts)
            if scores:
                rerank_used = True
                scored = []
                for (chunk, emb_score), rr_score in zip(candidates, scores):
                    meta = chunk.get("metadata", {})
                    meta["embedding_score"] = emb_score
                    meta["rerank_score"] = rr_score
                    scored.append((chunk, rr_score))
                reranked = sorted(scored, key=lambda x: x[1], reverse=True)
                # compute movement
                orig_order = [c.get("metadata", {}).get("chunk_id") for c, _ in candidates]
                new_order = [c.get("metadata", {}).get("chunk_id") for c, _ in reranked]
                rerank_changed = sum(1 for a, b in zip(orig_order, new_order) if a != b)
                rerank_table = [
                    f"{c.get('metadata',{}).get('doc_id')} emb={c.get('metadata',{}).get('embedding_score',0):.3f} rr={s:.3f} {c.get('text','')[:60].replace(chr(10),' ')}"
                    for c, s in reranked[:5]
                ]
                # append remainder unchanged
                filtered_hits = reranked + filtered_hits[top_n:]
    processing_chunk_matches: list[str] = []
    if processing_focus:
        processing_filtered = []
        dropped = 0
        range_regex = re.compile(r"\b\d{1,2}\s*-\s*\d{1,2}\s*(business\s+)?days?")
        for c, s in filtered_hits:
            text_lower = c.get("text", "").lower()
            match_keyword = None
            for kw in PROCESSING_CHUNK_KEYWORDS:
                if kw in text_lower:
                    match_keyword = kw
                    break
            if not match_keyword and range_regex.search(text_lower):
                match_keyword = "day_range"
            if match_keyword:
                processing_chunk_matches.append(f"{c.get('metadata',{}).get('chunk_id')}:{match_keyword}")
                processing_filtered.append((c, s))
            else:
                dropped += 1
        if debug:
            logger.info(
                "Processing chunk filter kept=%d dropped=%d matches=%s",
                len(processing_filtered),
                dropped,
                ",".join(processing_chunk_matches),
            )
        if processing_filtered:
            filtered_hits = processing_filtered
        else:
            fallback_used = True
            fallback_reasons.append("processing chunk filter relaxed (no keyword matches)")
    if processing_focus and processing_country and not filtered_hits:
        # fallback to original shortlist if filter was too strict
        filter_warning = True
        fallback_used = True
        fallback_reasons.append("processing country filter relaxed (no chunks)")
        filtered_hits = chunk_hits
    hits = filtered_hits[: min(rerank_top_k_val, POST_FILTER_MAX_CHUNKS)]

    top_score = hits[0][1] if hits else 0.0
    confidence = bool(hits) and top_score >= cfg.confidence_threshold
    filter_warning = filter_warning or fallback_used
    info = {
        "allowed_tags": sorted(allowed_tags),
        "intent": intent or "none",
        "intent_confidence": intent_conf,
        "intent_rule": intent_rule,
        "prefer_china_default": prefer_china_default,
        "filtered_applied": bool(allowed_tags),
        "filter_warning": filter_warning,
        "fallback_used": fallback_used,
        "fallback_reasons": fallback_reasons,
        "top_score": top_score,
        "shortlisted_docs": [doc for doc, _ in shortlisted],
        "processing_country": processing_country,
        "contact_strict": contact_strict,
        "excluded_docs": exclusions,
        "contact_allowed_docs": sorted(contact_allowed_docs),
        "contact_include_reasons": contact_include_reasons,
        "processing_chunk_matches": processing_chunk_matches,
        "rerank_enabled": rerank_enabled,
        "rerank_used": rerank_used,
        "rerank_model": reranker_model_name if rerank_enabled or rerank_used else None,
        "rerank_device": rerank_device,
        "rerank_top_n": rerank_top_n_val,
        "rerank_top_k": rerank_top_k_val,
        "rerank_changed": rerank_changed,
        "rerank_table": rerank_table,
    }
    if debug:
        logger.info(
            "Router intent=%s conf=%.2f rule=%s allowed_tags=%s shortlisted=%s top_score=%.3f",
            info["intent"],
            info["intent_confidence"],
            intent_rule,
            ",".join(sorted(allowed_tags)) or "none",
            ",".join(info["shortlisted_docs"]),
            top_score,
        )
    return hits, confidence, info
