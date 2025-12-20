from __future__ import annotations

import logging
import os
import re
from typing import Iterable, List, Tuple

from .config import Config, DEFAULT_CONFIG
from .embeddings import EmbeddingModel
from .index_store import IndexStore, SklearnVersionMismatch
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retrieval import _infer_countries, _tags_set, retrieve

logger = logging.getLogger(__name__)

# NOTE: Answering tuned for deterministic extractive mode: we honor router/debug info,
# enforce refusal when evidence is weak, and keep sources/context transparent.

# Refusal tuning
REFUSAL_SCORE_THRESHOLD = 0.25
NO_LLM_MAX_CHUNKS_DEFAULT = 6
NO_LLM_MAX_CHUNKS_PROCESSING_TIME = 2
CONTACT_CARD_KEYWORDS = [
    "street",
    "office",
    "floor",
    "opening hours",
    "hours",
    "mon",
    "fri",
    "whatsapp",
    "telegram",
    "wechat",
    "+7",
    "address",
    "find us",
    "yekaterinburg",
    "chelyabinsk",
]
CONTACT_EXCLUDE_KEYWORDS = ["bank", "invoice", "requisites", "payment", "account"]
PROCESSING_TIME_KEYWORDS = [
    "business day",
    "days",
    "urgent",
    "express",
    "processing",
    "processing time",
    "term",
]
PROCESSING_RERANK_MARGIN = 0.1


def _format_sources(hits: Iterable[Tuple[dict, float]]) -> List[str]:
    seen = set()
    sources = []
    for chunk, score in hits:
        meta = chunk.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        title = meta.get("title") or doc_id
        source_url = meta.get("source_url") or meta.get("local_path", "")
        sources.append(f"{title} ({source_url}) [score={score:.3f}]")
    return sources


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _select_sentences(hits: List[Tuple[dict, float]], question: str, max_lines: int = 8) -> List[str]:
    question_terms = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 2}
    selected = []
    seen = set()
    for chunk, _ in hits:
        sentences = _split_sentences(chunk.get("text", ""))
        chunk_picks = []
        for sent in sentences:
            tokens = {w.lower() for w in re.findall(r"\w+", sent)}
            if tokens & question_terms:
                chunk_picks.append(sent)
            if len(chunk_picks) >= 2:
                break
        if not chunk_picks and sentences:
            chunk_picks.append(sentences[0])
        for sent in chunk_picks:
            if sent not in seen:
                selected.append(sent)
                seen.add(sent)
            if len(selected) >= max_lines:
                return selected
    return selected[:max_lines]


def _extractive_answer(question: str, hits: List[Tuple[dict, float]], confidence: bool, top_k: int) -> str:
    if not hits:
        return "I can't find this in the provided sources. Please reach out via the listed contacts."
    sentences = _select_sentences(hits[:top_k], question, max_lines=8)
    lines: List[str] = []
    if not confidence:
        lines.append("I couldn't find an exact rule in the sources; please use handoff.")
    if sentences:
        lines.extend(f"- {s}" for s in sentences)
    else:
        lines.append("No matching sentences found in context.")
    return "\n".join(lines)


def _format_contact_card(question: str, hits: List[Tuple[dict, float]], allowed_docs: set[str]) -> str:
    q_lower = question.lower()
    allow_email = "email" in q_lower or "mail" in q_lower
    allow_bank = any(term in q_lower for term in ["bank", "invoice", "requisites"])
    lines: List[str] = []
    seen: set[str] = set()
    for chunk, _ in hits:
        meta = chunk.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        if allowed_docs and doc_id and doc_id not in allowed_docs:
            continue
        for raw_line in chunk.get("text", "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line_lower = line.lower()
            if "bank" in line_lower and not allow_bank:
                continue
            if not allow_email and ("@" in line_lower or "mail" in line_lower):
                continue
            if any(word in line_lower for word in CONTACT_EXCLUDE_KEYWORDS) and not allow_bank:
                continue
            if "furniture" in line_lower or "curtains" in line_lower:
                continue
            city_match = any(city in line_lower for city in ["yekaterinburg", "chelyabinsk"])
            if not any(key in line_lower for key in CONTACT_CARD_KEYWORDS) and not line.lstrip().startswith("+7"):
                if not city_match and not line_lower.endswith(":"):
                    continue
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
    if not lines:
        return "I couldn't find contact details in the sources; please call the main office."
    return "\n".join(lines[:12])


def _llm_answer(question: str, context: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        # Support openai>=1.0
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, context)},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("OpenAI call failed: %s", exc)
        return None


def _has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _select_evidence(
    hits: List[Tuple[dict, float]],
    intent: str,
    allowed_tags: set[str],
    contact_strict: bool,
    contact_allowed_docs: set[str],
    processing_country: str | None,
    fallback_used: bool,
    query_text: str,
    rerank_used: bool = False,
    debug: bool = False,
) -> List[Tuple[dict, float]]:
    if not hits:
        return hits

    def _is_contact_doc(meta: dict) -> bool:
        doc_id = meta.get("doc_id", "")
        tags = _tags_set(meta.get("tags"))
        if contact_allowed_docs:
            return doc_id in contact_allowed_docs
        return "contact" in tags or "locations" in tags or doc_id in contact_allowed_docs

    selected = hits
    q_lower = query_text.lower()

    if intent == "contact" and contact_strict:
        selected = [(c, s) for c, s in hits if _is_contact_doc(c.get("metadata", {}))]

    if intent == "processing_time":
        country_lock = processing_country == "china" or ("china" in allowed_tags and not fallback_used)

        def _is_processing_chunk(chunk: dict) -> bool:
            text_lower = chunk.get("text", "").lower()
            has_kw = any(kw in text_lower for kw in PROCESSING_TIME_KEYWORDS)
            has_num = _has_number(text_lower) or bool(re.search(r"\b\d{1,2}\s*-\s*\d{1,2}\b", text_lower))
            return has_kw and has_num

        filtered = []
        for c, s in selected:
            meta = c.get("metadata", {})
            tags = _tags_set(meta.get("tags"))
            countries = _infer_countries(meta)
            if country_lock and ({"japan", "taiwan"} & (tags | countries)) and not fallback_used:
                continue
            if _is_processing_chunk(c):
                filtered.append((c, s))
        if filtered:
            selected = filtered
        else:
            if debug:
                logger.warning("processing_time: no strong evidence, using fallback chunks")
        if rerank_used and len(selected) > 1:
            if selected[0][1] - selected[1][1] >= PROCESSING_RERANK_MARGIN:
                selected = selected[:1]

    # Deduplicate by doc and text signature
    deduped = []
    seen = set()
    for c, s in selected:
        meta = c.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        sig = (doc_id, c.get("text", "").strip()[:200].lower())
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append((c, s))
    selected = deduped

    max_chunks = NO_LLM_MAX_CHUNKS_PROCESSING_TIME if intent == "processing_time" else NO_LLM_MAX_CHUNKS_DEFAULT
    return selected[:max_chunks]


def answer_question(
    question: str,
    top_k: int | None = None,
    use_llm: bool = True,
    show_context: bool = False,
    config: Config | None = None,
    rebuild_on_mismatch: bool = False,
    debug: bool = False,
    rerank: bool = False,
    rerank_mode: str = "off",
    rerank_top_n: int | None = None,
    rerank_top_k: int | None = None,
    reranker_model: str | None = None,
) -> dict:
    cfg = config or DEFAULT_CONFIG
    index_store, embed_model, index_meta = IndexStore.load(
        cfg.index_dir, rebuild_on_mismatch=rebuild_on_mismatch
    )
    hits, confidence, info = retrieve(
        question,
        index_store,
        embed_model,
        top_k=top_k,
        config=cfg,
        debug=debug,
        rerank=rerank,
        rerank_mode=rerank_mode,
        rerank_top_n=rerank_top_n,
        rerank_top_k=rerank_top_k,
        reranker_model=reranker_model,
    )
    info["backend"] = getattr(embed_model, "backend", "unknown")
    info["model_name"] = getattr(embed_model, "model_name", None) or index_meta.get("embedding", {}).get(
        "model_name"
    )
    if debug and info.get("top_score", 0.0) > 1.0 + 1e-6:
        logger.warning("Top score %.4f exceeds cosine bound; check normalization.", info.get("top_score", 0.0))

    allowed_contact_docs = set(info.get("contact_allowed_docs") or [])
    hits_for_answer = hits
    if not use_llm:
        hits_for_answer = _select_evidence(
            hits,
            intent=info.get("intent", "none"),
            allowed_tags=set(info.get("allowed_tags") or []),
            contact_strict=info.get("contact_strict", False),
            contact_allowed_docs=allowed_contact_docs,
            processing_country=info.get("processing_country"),
            fallback_used=info.get("fallback_used", False),
            query_text=question,
            rerank_used=info.get("rerank_used", False),
            debug=debug,
        )

    context = "\n\n".join(f"[{chunk['metadata'].get('chunk_id')}] {chunk['text']}" for chunk, _ in hits_for_answer)
    answer_text = None
    if use_llm:
        answer_text = _llm_answer(question, context)

    top_score = info.get("top_score", 0.0)
    if not answer_text and info.get("contact_strict"):
        answer_text = _format_contact_card(question, hits_for_answer, allowed_contact_docs)
    if not answer_text:
        if top_score < REFUSAL_SCORE_THRESHOLD:
            answer_text = "I can't find this in the provided sources. Please contact the office for details."
            confidence = False
        else:
            answer_text = _extractive_answer(question, hits_for_answer, confidence, top_k or cfg.top_k)

    sources = _format_sources(hits_for_answer)
    context_lines = []
    if show_context and hits_for_answer:
        for chunk, score in hits_for_answer:
            meta = chunk.get("metadata", {})
            context_lines.append(
                f"- {meta.get('chunk_id')} (doc_id={meta.get('doc_id')} score={score:.3f}): {chunk.get('text','')[:400]}"
            )

    return {
        "answer_text": answer_text,
        "sources": sources,
        "hits": hits,
        "confidence": confidence,
        "retrieval_info": info,
        "context_lines": context_lines,
        "top_score": top_score,
    }
