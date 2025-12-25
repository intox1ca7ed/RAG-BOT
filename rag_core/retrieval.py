from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import logging
import os
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
POST_FILTER_LIMITS = {"default": 8, "delivery": 10, "contact": 8, "opening_hours": 8}
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
OPENING_HOURS_TERMS = [
    "opening hours",
    "open",
    "open on",
    "when are you open",
    "work on",
    "hours",
    "weekend",
    "saturday",
    "sunday",
    "sat",
    "sun",
]
DELIVERY_TERMS = [
    "send",
    "ship",
    "mail",
    "courier",
    "express",
    "passport",
    "documents",
    "delivery",
    "another city",
    "flying envelope",
    "s7",
    "ural airlines",
]
APOSTILLE_TERMS = [
    "apostille",
    "legalization",
    "legalisation",
    "notary",
    "certify",
    "stamp",
    "consular legalization",
]
LEGALIZATION_TERMS = [
    "legalization",
    "legalisation",
    "consular legalization",
    "consulate legalization",
    "embassy legalization",
]
MAIN_PAGE_TAGS = {"main_page", "overview", "general"}
DELIVERY_BOOST_TERMS = ["courier", "delivery", "send passport", "express mail", "flying envelope"]
CONTACT_BOOST_TERMS = ["phone", "address", "office", "contact", "number", "hours", "schedule"]
HOURS_BOOST_TERMS = ["monday", "tuesday", "weekday", "hours", "open", "close", "until", "from"]
CUSTOMS_TERMS = ["duty", "customs", "per kg", "tariff", "hs code"]
FURNITURE_TERMS = ["furniture", "tour"]
FURNITURE_COST_TERMS = ["cost", "price", "how much"]
MESSENGER_TERMS = ["wechat", "whatsapp", "telegram", "messenger", "phone", "email", "contact", "id", "handle"]
ROUTER_INTENTS = [
    "contact",
    "opening_hours",
    "delivery",
    "apostille",
    "legalization",
    "apostille_vs_legalization",
    "furniture_tour_cost",
    "customs_duties",
    "processing_time",
    "visa_requirement",
    "visa_price",
    "cargo_delivery",
    "form",
    "none",
]


def _tags_set(meta_tags: str | list | None) -> set[str]:
    if meta_tags is None:
        return set()
    if isinstance(meta_tags, list):
        items = meta_tags
    else:
        items = str(meta_tags).split(",")
    return {t.strip().lower() for t in items if t and str(t).strip()}


def _has_messenger_term(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in MESSENGER_TERMS)


def _merged_tags(meta: dict) -> set[str]:
    tags = _tags_set(meta.get("tags"))
    topic = meta.get("topic")
    section = meta.get("section")
    if topic:
        tags.add(str(topic).lower())
    if section:
        tags.add(str(section).lower())
    return tags


def _is_main_page(meta: dict) -> bool:
    doc_id = str(meta.get("doc_id", "")).lower()
    tags = _merged_tags(meta)
    return "main_page" in doc_id or bool(MAIN_PAGE_TAGS & tags)


def _infer_countries(meta: dict) -> set[str]:
    tags = _tags_set(meta.get("tags"))
    country_field = str(meta.get("country", "")).lower()
    doc_id = str(meta.get("doc_id", "")).lower()
    title = str(meta.get("title", "")).lower()
    local_path = str(meta.get("local_path", "")).lower()
    countries = set()
    if country_field in {"china", "japan", "taiwan"}:
        countries.add(country_field)
    for kw in ("china", "japan", "taiwan", "uae"):
        if kw in tags:
            countries.add(kw)
    if "uae" in doc_id or "uae" in title or "uae" in local_path:
        countries.add("uae")
    return countries


def _detect_allowed_tags(question: str) -> tuple[set[str], float]:
    q = question.lower()
    allowed = set()
    confidence = 0.0
    time_terms = bool(re.search(r"\b(how long|processing|time|days?)\b", q))
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
    if any(term in q for term in CUSTOMS_TERMS):
        allowed.add("customs_duties")
        confidence = max(confidence, 0.7)
    if any(term in q for term in FURNITURE_TERMS) and not any(term in q for term in CUSTOMS_TERMS):
        allowed.discard("customs_duties")
    if time_terms:
        confidence = min(confidence, 0.6)  # avoid over-strict tag gating for timing questions
    return allowed, confidence


def _detect_intent(question: str) -> tuple[str | None, float, str]:
    q = question.lower()
    contact_terms = ["contact", "office", "address", "phone", "email", "where", "find", "location", "wechat"]
    visa_req_terms = [
        "do i need a visa",
        "need a visa",
        "visa-free",
        "visa free",
        "entry conditions",
        "entry requirement",
        "right now",
        "can i enter",
        "how long can i stay",
        "stay",
        "30 days",
        "20 days",
    ]
    messenger_terms = ["wechat", "whatsapp", "telegram", "messenger", "phone", "email", "contact", "id", "handle"]
    cargo_terms = [
        "container",
        "containers",
        "20ft",
        "40ft",
        "cube",
        "cbm",
        "volume",
        "pallet",
        "shipping",
        "cargo",
        "freight",
        "weights",
        "kg",
        "dimensions",
        "size",
        "sizes",
    ]
    visa_price_cost_terms = ["how much", "cost", "price", "fee", "fees"]
    visa_codes = ["q2", "l", "m", "x1", "x2", "z"]
    form_terms = ["application form", "visa application form", "visa form", "anketa", "опросный лист"]
    if any(term in q for term in visa_req_terms):
        return "visa_requirement", 0.9, "visa_requirement_keywords"
    if any(term in q for term in form_terms):
        return "form", 0.85, "form_keywords"
    if any(term in q for term in messenger_terms):
        return "contact", 0.85, "contact_messenger"
    if any(code in q for code in visa_codes) and any(term in q for term in visa_price_cost_terms) and "visa" in q:
        return "visa_price", 0.9, "visa_price_code_cost"
    if "visa" in q and any(term in q for term in visa_price_cost_terms):
        return "visa_price", 0.86, "visa_price_keywords"
    if any(term in q for term in cargo_terms):
        return "cargo_delivery", 0.8, "cargo_keywords"
    if any(term in q for term in contact_terms):
        return "contact", 0.9, "contact_keywords"
    if any(term in q for term in OPENING_HOURS_TERMS):
        return "opening_hours", 0.9, "hours_keywords"
    if any(kw in q for kw in ["apostille vs", "apostille and legalization", "difference between apostille", "apostille or legalization"]):
        return "apostille_vs_legalization", 0.85, "apostille_vs_legalization_keywords"
    if any(term in q for term in DELIVERY_TERMS):
        return "delivery", 0.85, "delivery_keywords"
    has_apostille = any(term in q for term in APOSTILLE_TERMS)
    has_legalization = any(term in q for term in LEGALIZATION_TERMS)
    has_customs = any(term in q for term in CUSTOMS_TERMS)
    if any(term in q for term in FURNITURE_TERMS) and any(term in q for term in FURNITURE_COST_TERMS) and not has_customs:
        return "furniture_tour_cost", 0.8, "furniture_cost_keywords"
    if has_customs:
        return "customs_duties", 0.8, "customs_keywords"
    if has_apostille and not ("consular" in q or "embassy" in q):
        return "apostille", 0.92, "apostille_keywords"
    if has_legalization:
        return "legalization", 0.8, "legalization_keywords"
    proc_match = re.search(r"\b(how long|processing|business\s+days?|days?)\b", q)
    if proc_match:
        return "processing_time", 0.8, f"processing_terms:{proc_match.group(1)}"
    if "form" in q or "application" in q:
        return "form", 0.6, "form_keyword"
    return None, 0.0, "none"


def _is_entry_conditions_query(q_lower: str) -> bool:
    return any(
        term in q_lower
        for term in [
            "entry condition",
            "entry conditions",
            "entry requirement",
            "entry requirements",
            "visa-free",
            "visa free",
            "right now",
            "how long can i stay",
            "stay",
        ]
    )


@dataclass
class RouterResult:
    intent: str | None
    intent_confidence: float
    allowed_tags: set[str]
    allowed_confidence: float
    rule: str
    backend: str
    prefer_china_default: bool = False
    rationale: str | None = None


def _rules_router(question: str) -> RouterResult:
    allowed_tags, allowed_conf = _detect_allowed_tags(question)
    intent, intent_conf, intent_rule = _detect_intent(question)
    prefer_china_default = False
    if "visa" in allowed_tags and not ({"china", "japan", "taiwan"} & allowed_tags):
        allowed_tags.add("china")
        prefer_china_default = True
        allowed_conf = max(allowed_conf, 0.8)
    if intent == "contact" and intent_conf >= STRICT_FILTER_CONFIDENCE:
        allowed_tags = {"contact", "locations", "hours"}
        allowed_conf = 1.0
    if intent == "visa_requirement":
        allowed_tags = {"visa", "china"}
        allowed_conf = max(allowed_conf, 0.85)
    if intent == "visa_price":
        allowed_tags = {"visa", "china"}
        allowed_conf = max(allowed_conf, 0.85)
    if intent == "apostille_vs_legalization":
        allowed_tags = {"apostille", "legalization"}
        allowed_conf = max(allowed_conf, 0.85)
    if intent == "cargo_delivery":
        allowed_tags = {"cargo", "delivery"}
        allowed_conf = max(allowed_conf, 0.8)
    if intent == "form":
        if "china" in question.lower():
            allowed_tags = {"form", "china"}
        else:
            allowed_tags = {"form"}
        allowed_conf = max(allowed_conf, 0.8)
    return RouterResult(
        intent=intent,
        intent_confidence=intent_conf,
        allowed_tags=allowed_tags,
        allowed_confidence=allowed_conf,
        rule=intent_rule,
        backend="rules",
        prefer_china_default=prefer_china_default,
    )


def _is_rules_confident(
    router: RouterResult,
    query: str,
    debug: bool = False,
    filter_warning: bool = False,
    top_doc_score: float | None = None,
    top2_doc_score: float | None = None,
) -> bool:
    reasons: list[str] = []
    confident = (
        router.intent_confidence >= 0.8
        and bool(router.allowed_tags)
        and len(router.allowed_tags) <= 3
        and router.rule not in {"fallback", "generic"}
    )
    if filter_warning:
        confident = False
        reasons.append("filter_warning")
    if top_doc_score is not None and top2_doc_score is not None:
        margin = top_doc_score - top2_doc_score
        if margin < 0.08:
            confident = False
            reasons.append(f"margin_low:{margin:.3f}")
    brittle = False
    if router.intent in {"processing_time", "delivery"}:
        brittle = True
    if router.rule.startswith("processing_terms") or router.rule.startswith("delivery_keywords"):
        brittle = True
    competing_signals = [
        "do i need a visa",
        "visa-free",
        "visa free",
        "entry",
        "stay",
        "how long can i stay",
        "difference between",
        "which is needed",
        "legalization vs apostille",
        "apostille vs legalization",
    ]
    if brittle and any(term in query.lower() for term in competing_signals):
        confident = False
        reasons.append("competing_signal")

    if debug:
        logger.info(
            "router_auto_decision=%s intent=%s conf=%.2f allowed_tags=%s rule=%s reasons=%s",
            "rules_confident" if confident else "rules_not_confident",
            router.intent,
            router.intent_confidence,
            ",".join(sorted(router.allowed_tags)) or "none",
            router.rule,
            ",".join(reasons) or "none",
        )
    return confident


def _build_doc_catalog_summary(chunks: List[dict], limit: int = 30) -> list[dict]:
    seen = set()
    summary: list[dict] = []
    for chunk in chunks:
        meta = chunk.get("metadata", {}) or {}
        doc_id = meta.get("doc_id")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        tags = list(_merged_tags(meta))
        summary.append(
            {
                "doc_id": doc_id,
                "title": meta.get("title") or "",
                "tags": tags,
                "topic": meta.get("topic") or "",
                "country": meta.get("country") or "",
            }
        )
        if len(summary) >= limit:
            break
    return summary


def llm_route(
    question: str,
    available_intents: list[str],
    available_tags: list[str],
    doc_catalog_summary: list[dict],
    llm_model: str,
    llm_timeout: int | None = None,
    debug: bool = False,
) -> RouterResult | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    client = OpenAI(api_key=api_key)
    system_msg = (
        "You are a routing assistant. Choose the best intent and allowed_tags for a RAG system.\n"
        "Valid intents: " + ", ".join(available_intents) + ".\n"
        "Valid tags: " + ", ".join(available_tags) + ".\n"
        "Routing cheatsheet:\n"
        "- do I need a visa / visa-free / stay X days / entry conditions -> visa_requirement\n"
        "- processing time / urgent / days to prepare visa -> processing_time\n"
        "- send passport / courier / express mail -> delivery\n"
        "- container volume / kg / weights / shipping / cargo -> cargo_delivery\n"
        "- wechat / whatsapp / telegram / phone / email / contact / id / handle -> contact\n"
        "- apostille vs legalization / difference -> apostille_vs_legalization (or apostille)\n"
        "Choose ONLY from the provided intents; if uncertain choose 'none' with low confidence. Return at most 3 tags.\n"
    )
    catalog_text = "\n".join(
        f"- {d.get('doc_id','')}: title='{d.get('title','')}', tags={d.get('tags',[])}, topic='{d.get('topic','')}', country='{d.get('country','')}'"
        for d in doc_catalog_summary
    )
    user_msg = (
        f"Question: {question}\n\nDoc catalog (metadata only):\n{catalog_text}\n\n"
        "Return a JSON object with keys: intent (string), allowed_tags (array of strings), confidence (0-1 float), rationale (short string). "
        "Use only intents/tags from the provided lists. If uncertain, use intent='none', allowed_tags=[], confidence<=0.5. Do not return more than 3 tags. "
        "If the question is asking how to reach us or for messenger IDs (wechat/whatsapp/telegram/phone/email/contact/id/handle), choose intent='contact'."
    )

    def _call():
        kwargs = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 200,
            "temperature": 0.0,
        }
        if llm_timeout is not None:
            kwargs["timeout"] = llm_timeout
        return client.chat.completions.create(**kwargs)

    response = None
    for attempt in range(2):
        try:
            response = _call()
        except Exception as exc:
            if attempt == 0:
                user_msg_retry = user_msg + "\nReturn only JSON on the next attempt."
                user_msg = user_msg_retry
                continue
            logger.warning("llm_router_failed: %s", exc)
            return None
        if response:
            content = response.choices[0].message.content.strip()
            import json

            def _parse(text: str):
                try:
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        text = text[start : end + 1]
                    return json.loads(text)
                except Exception:
                    return None

            parsed = _parse(content)
            if parsed:
                allowed_tags = [t.strip().lower() for t in parsed.get("allowed_tags", []) if t]
                intent = parsed.get("intent") or None
                try:
                    conf = float(parsed.get("confidence", 0.0) or 0.0)
                except Exception:
                    conf = 0.5
                rationale = parsed.get("rationale") or ""
                corrections: list[str] = []
                if intent not in available_intents:
                    intent = "none"
                    conf = 0.0
                    corrections.append("intent_reset")
                allowed_tags = [t for t in allowed_tags if t in set(available_tags)][:3]
                if len(allowed_tags) < len(parsed.get("allowed_tags", [])):
                    corrections.append("tags_trimmed")
                if intent == "none" and _has_messenger_term(question):
                    intent = "contact"
                    conf = max(conf, 0.6)
                    corrections.append("override_contact_messenger")
                if debug and corrections:
                    logger.info("llm_router_output_corrected=%s", ",".join(corrections))
                return RouterResult(
                    intent=intent,
                    intent_confidence=conf,
                    allowed_tags=set(allowed_tags),
                    allowed_confidence=conf,
                    rule="llm_router",
                    backend="llm",
                    rationale=rationale,
                )
        # retry once if parse failed
        user_msg = user_msg + "\nReturn only JSON."
    return None


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
    entry_conditions: bool,
    allow_main_page: bool,
    query_text: str,
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

        is_main_page = _is_main_page(meta)
        is_duties = "duties" in tags or "customs" in tags or "customs_duties" in tags or "duties" in doc_id

        if contact_strict:
            if doc_id not in contact_allowed_docs:
                exclusions.append(f"{doc_id}: excluded (contact strict allowlist)")
                continue
            include_reasons.append(f"{doc_id}: included (contact allowlist)")

        if intent == "opening_hours":
            if not (is_main_page or "contact" in tags or doc_id in contact_doc_ids):
                exclusions.append(f"{doc_id}: excluded (opening_hours allowlist)")
                continue

        if intent == "delivery":
            if not (is_main_page or any(k in tags for k in {"delivery", "shipping", "express"})):
                exclusions.append(f"{doc_id}: excluded (delivery allowlist)")
                continue

        if intent == "apostille":
            if not (is_main_page or ({"apostille", "legalization", "legalisation"} & tags)):
                exclusions.append(f"{doc_id}: excluded (apostille allowlist)")
                continue
            other_countries = {"japan", "taiwan", "uae"} & (countries | tags)
            if other_countries and "uae" not in query_text.lower():
                exclusions.append(f"{doc_id}: excluded (apostille other country tags: {','.join(sorted(other_countries))})")
                continue
        if intent == "legalization":
            if not (is_main_page or {"legalization", "legalisation"} & tags or "legalization" in doc_id):
                exclusions.append(f"{doc_id}: excluded (legalization allowlist)")
                continue
        if intent == "furniture_tour_cost":
            if is_duties:
                exclusions.append(f"{doc_id}: excluded (duties/customs for furniture cost)")
                continue
        if intent == "customs_duties":
            if not is_duties and not ("customs" in tags or "duties" in tags):
                exclusions.append(f"{doc_id}: excluded (customs duties intent prefers duties docs)")
                continue

        if processing_country:
            if not (processing_country in countries or processing_country in tags or is_main_page):
                exclusions.append(f"{doc_id}: excluded (processing intent country mismatch)")
                continue
            if not ({"visa", "processing", "main_page", "general"} & tags):
                if not is_main_page:
                    exclusions.append(f"{doc_id}: excluded (processing intent missing visa/processing/main_page tag)")
                    continue
            other_countries = {"japan", "taiwan"} & (countries | tags)
            if other_countries:
                if not is_main_page:
                    exclusions.append(f"{doc_id}: excluded (processing intent other country tags: {','.join(sorted(other_countries))})")
                    continue

        if exclude_furniture and not query_has_furniture and (FURNITURE_TRAP_TAGS & tags):
            exclusions.append(f"{doc_id}: excluded (furniture/tour trap)")
            continue

        if allow_main_page and is_main_page:
            include_reasons.append(f"{doc_id}: included (main_page override)")
        elif not contact_strict and not _doc_matches_filters(allowed_tags, allowed_conf, countries, tags, prefer_china_default):
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
        if intent == "processing_time":
            day_regex = re.compile(r"\b(business\s+days?|days?)\b")
            for idx in idxs:
                text_lower = chunks[idx].get("text", "").lower()
                if day_regex.search(text_lower):
                    score += PROCESSING_TIME_BOOST
                    break
        if intent == "apostille" and is_main_page:
            score += 0.3
        if entry_conditions and is_main_page:
            score += 0.25
        if intent == "furniture_tour_cost" and is_duties:
            score -= 0.35
        if intent == "customs_duties" and is_duties:
            score += 0.35
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
    router_backend: str = "rules",
    llm_model: str = "gpt-4o-mini",
    llm_timeout: int | None = 60,
    _auto_override: bool = False,
) -> tuple[List[tuple[dict, float]], bool, dict]:
    cfg = config or DEFAULT_CONFIG
    k = top_k or cfg.top_k
    rerank_top_n_val = rerank_top_n or 30
    rerank_top_k_val = rerank_top_k or k
    router_backend_mode = (router_backend or "rules").lower()
    auto_mode = router_backend_mode == "auto"
    rules_router = _rules_router(question)
    selected_router = rules_router
    used_backend = "rules"
    llm_rationale = None
    available_tags_set = set()
    for chunk in index_store.chunks:
        meta = chunk.get("metadata", {}) or {}
        available_tags_set |= _merged_tags(meta)
        if "country" in meta and meta.get("country"):
            available_tags_set.add(str(meta.get("country")).lower())
    doc_catalog_summary = _build_doc_catalog_summary(index_store.chunks)
    if router_backend_mode == "llm":
        llm_result = llm_route(
            question,
            available_intents=ROUTER_INTENTS,
            available_tags=sorted(available_tags_set) or list(rules_router.allowed_tags),
            doc_catalog_summary=doc_catalog_summary,
            llm_model=llm_model,
            llm_timeout=llm_timeout,
            debug=debug,
        )
        if llm_result:
            selected_router = llm_result
            used_backend = "llm"
            llm_rationale = llm_result.rationale
            if debug and llm_rationale:
                logger.info("llm_router_rationale=%s", llm_rationale)
        else:
            if debug:
                logger.warning("llm_router_failed_fallback_to_rules=true")
            selected_router = rules_router
            used_backend = "rules"

    router_result = selected_router
    allowed_tags = set(router_result.allowed_tags)
    allowed_conf = router_result.allowed_confidence
    intent = router_result.intent
    intent_conf = router_result.intent_confidence
    intent_rule = router_result.rule
    prefer_china_default = router_result.prefer_china_default
    if "visa" in allowed_tags and not ({"china", "japan", "taiwan"} & allowed_tags):
        # Ensure china default also applied to llm routing results
        allowed_tags.add("china")
        prefer_china_default = True
        allowed_conf = max(allowed_conf, 0.8)
    processing_focus = intent == "processing_time"
    operational_intent = intent in {
        "processing_time",
        "opening_hours",
        "delivery",
        "apostille",
        "legalization",
        "apostille_vs_legalization",
    }
    if intent == "contact" and intent_conf >= STRICT_FILTER_CONFIDENCE:
        allowed_tags = {"contact", "locations", "hours"}
        allowed_conf = 1.0
    # Processing-time intent country restriction
    processing_country: str | None = None
    q_lower = question.lower()
    entry_conditions = intent == "visa_requirement" and _is_entry_conditions_query(q_lower)
    payment_query = "pay" in q_lower or "payment" in q_lower
    if processing_focus and ("china" in allowed_tags or any(term in q_lower for term in PROCESSING_COUNTRY_TERMS)):
        processing_country = "china"

    contact_strict = intent == "contact" and intent_conf >= CONTACT_STRICT_CONFIDENCE
    consulate_ok = "consulate" in q_lower or "embassy" in q_lower
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
            if not consulate_ok and "consulate" in doc_id_lower:
                exclusions.append(f"{doc_id}: excluded (contact strict consulate filtered)")
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
        entry_conditions=entry_conditions,
        allow_main_page=operational_intent or entry_conditions,
        query_text=question,
    )

    filter_warning = False
    fallback_used = False
    fallback_reasons: list[str] = []
    allow_filter_fallback = not contact_strict and intent != "opening_hours"
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
            entry_conditions=entry_conditions,
            allow_main_page=operational_intent or entry_conditions,
            query_text=question,
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
            entry_conditions=entry_conditions,
            allow_main_page=operational_intent or entry_conditions,
            query_text=question,
        )

    shortlisted = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:DOC_SHORTLIST]
    shortlisted_ids = {doc_id for doc_id, _ in shortlisted}
    if processing_focus:
        for doc_id in doc_indices.keys():
            if "main_page" in doc_id or "main-page" in doc_id:
                shortlisted_ids.add(doc_id)
    if entry_conditions:
        for doc_id in doc_indices.keys():
            if "main_page" in doc_id or "main-page" in doc_id:
                shortlisted_ids.add(doc_id)

    def _chunk_allowed(meta: dict) -> bool:
        tags = _tags_set(meta.get("tags"))
        countries = _infer_countries(meta)
        doc_id_val = meta.get("doc_id", "")
        is_main = _is_main_page(meta)
        is_duties_chunk = "duties" in tags or "customs" in tags or "customs_duties" in tags or "duties" in doc_id_val
        if contact_strict:
            if contact_allowed_docs:
                return doc_id_val in contact_allowed_docs
            return ("contact" in tags) or doc_id_val in CONTACT_DOC_IDS
        if processing_country:
            if not (processing_country in tags or processing_country in countries or is_main):
                return False
            if {"japan", "taiwan"} & (countries | tags):
                if not is_main:
                    return False
            if not ({"visa", "processing", "main_page", "general"} & tags):
                if not is_main:
                    return False
        if intent in {"opening_hours"}:
            if not (is_main or "contact" in tags or "hours" in tags):
                return False
        if intent in {"delivery", "cargo_delivery"}:
            if not (is_main or any(k in tags for k in {"delivery", "shipping", "express", "mail", "cargo"})):
                return False
        if intent == "apostille":
            if not (is_main or ({"apostille", "legalization", "legalisation"} & tags)):
                return False
        if intent == "apostille_vs_legalization":
            if not (is_main or ({"apostille", "legalization", "legalisation"} & tags) or "legalization" in doc_id_val):
                return False
        if intent == "furniture_tour_cost" and is_duties_chunk:
            return False
        if intent == "customs_duties" and not is_duties_chunk:
            return False
        if exclude_furniture and not ("furniture" in q_lower or "tour" in q_lower):
            if FURNITURE_TRAP_TAGS & tags:
                return False
        return True

    # Gather chunk scores inside shortlisted docs
    vectors = index_store.vectors
    chunks = index_store.chunks
    chunk_hits: list[tuple[dict, float]] = []
    delivery_debug: list[str] = []
    for doc_id in shortlisted_ids:
        for idx in doc_indices.get(doc_id, []):
            sim = float(vectors[idx] @ query_vec)
            sim = max(-1.0, min(1.0, sim))
            meta = chunks[idx].get("metadata", {})
            text_lower = chunks[idx].get("text", "").lower()
            tags = _tags_set(meta.get("tags"))
            if intent == "contact" and ("contact" in tags or "locations" in tags or "hours" in tags):
                sim += 0.2
            if intent == "form" and ("form" in tags or meta.get("doc_type") == "csv"):
                sim += 0.05
            if processing_focus:
                if "day" in text_lower and "time" in text_lower:
                    sim += 0.3
            if intent == "delivery" and any(term in text_lower for term in DELIVERY_BOOST_TERMS):
                sim += 0.25
            if intent == "form":
                if "category of visa requested" in text_lower:
                    sim += 0.35
                elif "visa application form" in text_lower or "application form" in text_lower:
                    sim += 0.2
            if intent == "visa_requirement" and entry_conditions:
                if "visa-free" in text_lower and ("30 days" in text_lower or "30 day" in text_lower):
                    sim += 0.3
            if payment_query and "payment upon visa readiness" in text_lower:
                sim += 0.35
            if intent == "contact" and any(term in text_lower for term in CONTACT_BOOST_TERMS):
                sim += 0.25
            if intent == "opening_hours" and any(term in text_lower for term in HOURS_BOOST_TERMS):
                sim += 0.25
            sim = max(-1.0, min(1.0, sim))
            chunk_hits.append((chunks[idx], sim))
            if intent == "delivery" and _is_main_page(meta):
                delivery_debug.append(
                    f"{meta.get('chunk_id')} doc={meta.get('doc_id')} emb_sim={sim:.3f} delivery_candidate={'yes' if any(k in text_lower for k in ['express','flying envelope','s7','ural']) else 'no'}"
                )

    chunk_hits.sort(key=lambda x: x[1], reverse=True)
    # Post-filter chunks to enforce country/topic constraints and limit context size
    filtered_hits = [(c, s) for c, s in chunk_hits if _chunk_allowed(c.get("metadata", {}))]
    if intent == "cargo_delivery" and not filtered_hits:
        cargo_terms = ["cargo", "container", "containers", "shipping", "delivery"]
        broadened = []
        for c, s in chunk_hits:
            meta = c.get("metadata", {})
            doc_id = str(meta.get("doc_id", "")).lower()
            title = str(meta.get("title", "")).lower()
            topic = str(meta.get("topic", "")).lower()
            tags = _tags_set(meta.get("tags"))
            if any(term in doc_id for term in cargo_terms) or any(term in title for term in cargo_terms):
                broadened.append((c, s))
                continue
            if any(term in topic for term in cargo_terms):
                broadened.append((c, s))
                continue
            if tags & set(cargo_terms):
                broadened.append((c, s))
        if broadened:
            if debug:
                logger.info("cargo_delivery_filter_broadened=true matches=%d", len(broadened))
            filtered_hits = broadened

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
    limit = POST_FILTER_LIMITS.get(intent, POST_FILTER_LIMITS["default"])
    if limit is None or limit == 0:
        hits = filtered_hits
    else:
        hits = filtered_hits[: min(rerank_top_k_val, limit)]

    top_score = hits[0][1] if hits else 0.0
    top_doc_score = shortlisted[0][1] if shortlisted else 0.0
    top2_doc_score = shortlisted[1][1] if len(shortlisted) > 1 else 0.0
    confidence = bool(hits) and top_score >= cfg.confidence_threshold
    filter_warning = filter_warning or fallback_used
    rules_confident_flag = _is_rules_confident(
        rules_router,
        question,
        debug=debug,
        filter_warning=filter_warning,
        top_doc_score=top_doc_score,
        top2_doc_score=top2_doc_score,
    )
    if router_backend_mode == "auto" and used_backend == "rules" and not rules_confident_flag and not _auto_override:
        if debug:
            logger.info("router_auto_switch_to_llm=true")
        return retrieve(
            question,
            index_store,
            embed_model,
            top_k=top_k,
            config=config,
            debug=debug,
            rerank=rerank,
            rerank_mode=rerank_mode,
            rerank_top_n=rerank_top_n,
            rerank_top_k=rerank_top_k,
            reranker_model=reranker_model,
            router_backend="llm",
            llm_model=llm_model,
            llm_timeout=llm_timeout,
            _auto_override=True,
        )

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
        "top_doc_score": top_doc_score,
        "top2_doc_score": top2_doc_score,
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
        "delivery_debug": delivery_debug if intent == "delivery" else [],
        "router_backend": used_backend,
        "router_backend_requested": router_backend_mode,
        "router_rationale": llm_rationale,
        "router_rules_confident": rules_confident_flag,
    }
    original_result = (hits, confidence, info)

    messenger_guard = (
        auto_mode
        and intent == "contact"
        and intent_conf >= 0.8
        and _has_messenger_term(question)
    )

    if auto_mode and used_backend == "rules" and not rules_confident_flag and not _auto_override:
        llm_hits, llm_confidence, llm_info = retrieve(
            question,
            index_store,
            embed_model,
            top_k=top_k,
            config=config,
            debug=debug,
            rerank=rerank,
            rerank_mode=rerank_mode,
            rerank_top_n=rerank_top_n,
            rerank_top_k=rerank_top_k,
            reranker_model=reranker_model,
            router_backend="llm",
            llm_model=llm_model,
            llm_timeout=llm_timeout,
            _auto_override=True,
        )
        llm_intent = llm_info.get("intent")
        llm_conf = llm_info.get("intent_confidence", 0.0)
        if messenger_guard and (llm_intent != "contact" and llm_conf < 0.6):
            if debug:
                logger.info("auto_guardrail_applied=contact_messenger_prefer_rules")
        else:
            return llm_hits, llm_confidence, llm_info
    if debug:
        logger.info(
            "Router intent=%s conf=%.2f rule=%s backend=%s allowed_tags=%s shortlisted=%s top_score=%.3f",
            info["intent"],
            info["intent_confidence"],
            intent_rule,
            used_backend,
            ",".join(sorted(allowed_tags)) or "none",
            ",".join(info["shortlisted_docs"]),
            top_score,
        )
        logger.info(
            "Router trace backend_requested=%s selected_backend=%s rules_confident=%s shortlist=%d top_doc_score=%.3f top2_doc_score=%.3f filter_warning=%s fallback_reasons=%s",
            router_backend_mode,
            used_backend,
            rules_confident_flag,
            len(shortlisted),
            top_score,
            info.get("top2_doc_score", 0.0),
            filter_warning,
            ";".join(fallback_reasons) or "none",
        )
    return hits, confidence, info
