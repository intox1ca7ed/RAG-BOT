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
from .retrieval import DELIVERY_TERMS as RETRIEVAL_DELIVERY_TERMS
from .routing_log import append_routing_event

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
HOURS_KEYWORDS = [
    "opening hours",
    "hours",
    "mon",
    "fri",
    "sat",
    "sun",
    "saturday",
    "sunday",
    "weekend",
    "open",
]
DELIVERY_KEYWORDS = [
    "express",
    "courier",
    "mail",
    "send",
    "delivery",
    "flying envelope",
    "s7",
    "ural airlines",
    "passport",
    "documents",
    "another city",
    "ship",
    "post",
]
APOSTILLE_PRICE_PATTERN = re.compile(r"\b\d{1,3}(?:[ ,]\d{3})*\s*rub", re.IGNORECASE)
APOSTILLE_DAYS_PATTERN = re.compile(r"\b\d+\s*(working\s+)?days?", re.IGNORECASE)
CITATION_PATTERN = re.compile(r"[Ss](\d+)(?:\s*-\s*(\d+))?")


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
    city_only = {"chelyabinsk", "yekaterinburg", "royal tour"}
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
            if line_lower in city_only:
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


def _format_delivery_answer(hits: List[Tuple[dict, float]], query: str) -> str:
    if not hits:
        return "You can send your documents by express mail or flying envelope. Processing starts after we receive them."
    lines: List[str] = []
    note_added = False
    for chunk, _ in hits:
        text = chunk.get("text", "")
        text_lower = text.lower()
        if _is_delivery_chunk(text_lower):
            parts = []
            if "express" in text_lower or "mail" in text_lower or "courier" in text_lower:
                parts.append("express mail/courier")
            if "flying envelope" in text_lower or "s7" in text_lower or "ural" in text_lower:
                parts.append("flying envelope via S7 or Ural Airlines")
            if parts:
                lines.append("You can send documents by " + " and ".join(parts) + ".")
            if "receive" in text_lower or "office" in text_lower or "moscow" in text_lower:
                lines.append("Processing starts after documents arrive at our Moscow office.")
                note_added = True
    if not lines:
        # fallback: use first chunk text trimmed
        return hits[0][0].get("text", "")[:300]
    if not note_added:
        lines.append("Processing starts after we receive your documents.")
    return "\n".join(lines[:3])


def truncate_chunk(text: str, max_chars: int = 1200, debug: bool = False) -> str:
    if len(text) <= max_chars:
        return text
    head_len = max_chars // 2
    tail_len = max_chars - head_len - 5
    if debug:
        logger.info("Truncating context chunk from %d to %d chars", len(text), max_chars)
    return text[:head_len] + "\n...\n" + text[-tail_len:]


def _extract_citation_ids(text: str) -> List[int]:
    ids: List[int] = []
    for match in CITATION_PATTERN.finditer(text or ""):
        start = int(match.group(1))
        end_group = match.group(2)
        if end_group and end_group.isdigit():
            end = int(end_group)
            if end >= start:
                ids.extend(range(start, end + 1))
                continue
        ids.append(start)
    return ids


def _strip_invalid_citations(text: str, excerpt_count: int) -> str:
    if not text:
        return text

    def _repl(match: re.Match[str]) -> str:
        start = int(match.group(1))
        end_group = match.group(2)
        ids = [start]
        if end_group and end_group.isdigit() and int(end_group) >= start:
            ids = list(range(start, int(end_group) + 1))
        if all(1 <= i <= excerpt_count for i in ids):
            return match.group(0)
        return ""

    cleaned = CITATION_PATTERN.sub(_repl, text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def validate_llm_grounding(output_text: str, excerpt_count: int, mode: str) -> tuple[bool, set[int], str]:
    """
    Validate grounding by checking for excerpt references depending on mode.
    Returns (ok, used_ids, reason).
    """
    mode_normalized = (mode or "strict").lower()
    text = (output_text or "").strip()
    if mode_normalized == "none":
        return (bool(text), set(), "mode_none" if text else "empty_output")
    if not text:
        return False, set(), "empty_output"
    if excerpt_count <= 0:
        if mode_normalized == "strict":
            return False, set(), "no_excerpts"
        return True, set(), "no_excerpts"

    used_ids: List[int] = []
    invalid = False
    for cid in _extract_citation_ids(text):
        if 1 <= cid <= excerpt_count:
            used_ids.append(cid)
        else:
            invalid = True

    used_set = set(used_ids)
    if mode_normalized == "strict":
        if invalid:
            return False, used_set, "invalid_citation"
        if not used_set:
            return False, used_set, "no_citations"
        return True, used_set, "ok"

    if mode_normalized == "loose":
        if invalid:
            return False, used_set, "invalid_citation"
        if not used_set:
            return True, used_set, "no_citations"
        return True, used_set, "ok"

    return True, used_set, "ok"


def _llm_answer(
    question: str,
    context: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 600,
    llm_timeout: int | None = None,
    debug: bool = False,
    intent: str | None = None,
    grounding: str = "strict",
    excerpts: list[tuple[str, str]] | None = None,
    stricter: bool = False,
) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    grounding_mode = (grounding or "strict").lower()
    strict_flag = grounding_mode == "strict"
    loose_flag = grounding_mode == "loose"
    intent_line = ""
    if intent == "contact":
        intent_line = "Focus on address, hours, and phone numbers."
    elif intent == "delivery":
        intent_line = "Explain how to send documents (express mail, flying envelope, S7/Ural Airlines) and when processing starts."
    elif intent == "apostille":
        intent_line = "Include apostille price and processing time."
    elif intent == "processing_time":
        intent_line = "State processing time clearly."
    elif intent == "furniture_tour_cost":
        intent_line = "State tour price and what it includes, avoid customs duties."

    sys_msg = (
        "You are a customer-support QA bot. Answer only using the provided context excerpts. "
        "Do NOT invent facts. If the context does not answer, say so briefly and ask one clarifying question. "
        + intent_line
    )
    user_msg_lines = [f"Question: {question}", "", "Context excerpts:"]
    excerpt_ids = []
    if excerpts:
        for idx, (chunk_id, txt) in enumerate(excerpts, 1):
            sid = f"S{idx}"
            excerpt_ids.append(sid)
            user_msg_lines.append(f"[{sid}] {chunk_id}: {txt}")
    else:
        user_msg_lines.append(context)
    if strict_flag:
        user_msg_lines.append(
            "\nUse the excerpt IDs (S1, S2, ...) when citing evidence, e.g. [S1] or 'Sources used: S1'. Provide at least one reference from the provided excerpts. If the context does not answer, say so briefly."
        )
        if stricter:
            user_msg_lines.append(
                "Your previous response lacked valid citations. Add at least one correct excerpt ID from the provided context or state that the context is insufficient."
            )
    elif loose_flag:
        user_msg_lines.append(
            "\nAnswer using the provided excerpts; if unsure, say you can't find it. Citations are optional, but if you include them they must use the provided excerpt IDs (S1, S2, ...)."
        )
        if stricter:
            user_msg_lines.append("Remove any invalid citations or replace them with the correct excerpt IDs.")
    else:
        user_msg_lines.append("\nProvide a concise answer.")

    user_content = "\n".join(user_msg_lines)
    try:
        # Support openai>=1.0
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_content},
            ],
        }
        if llm_timeout is not None:
            kwargs["timeout"] = llm_timeout
        if debug:
            logger.info(
                "llm_model=%s max_tokens=%d temp=%.2f grounding=%s",
                model,
                max_tokens,
                temperature,
                grounding_mode,
            )

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as exc:
        if debug:
            logger.warning("LLM call failed: %s", exc)
        logger.warning("OpenAI call failed: %s", exc)
        return None


def _parse_sources_line(text: str, available_ids: set[str]) -> tuple[bool, set[str]]:
    # Deprecated; kept for backward compatibility if imported elsewhere.
    return False, set()


def _has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _is_delivery_chunk(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in DELIVERY_KEYWORDS)


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

    if intent == "opening_hours":
        hours_filtered = []
        pattern = re.compile(r"\b(mon|tue|wed|thu|fri|sat|sun)\b|\b\d{1,2}[:.]\d{2}", re.IGNORECASE)
        for c, s in selected:
            text_lower = c.get("text", "").lower()
            if pattern.search(text_lower) or any(kw in text_lower for kw in HOURS_KEYWORDS):
                hours_filtered.append((c, s))
        if hours_filtered:
            selected = hours_filtered[:2]

    if intent == "delivery":
        delivery_filtered = []
        for c, s in selected:
            text_lower = c.get("text", "").lower()
            if _is_delivery_chunk(text_lower):
                delivery_filtered.append((c, s))
        if delivery_filtered:
            selected = delivery_filtered[:2]
        else:
            generic = []
            for c, s in selected:
                text_lower = c.get("text", "").lower()
                if "passport" in text_lower or "documents" in text_lower:
                    generic.append((c, s))
            if generic:
                selected = generic[:2]
            elif debug:
                logger.warning("delivery intent: no delivery-specific chunks, falling back to generic main_page text")

    if intent == "apostille":
        apo_filtered = []
        for c, s in selected:
            text_lower = c.get("text", "").lower()
            if (APOSTILLE_PRICE_PATTERN.search(text_lower) or "rubles" in text_lower) and APOSTILLE_DAYS_PATTERN.search(
                text_lower
            ):
                apo_filtered.append((c, s))
        if apo_filtered:
            selected = apo_filtered[:2]

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
    llm_model: str = "gpt-4o-mini",
    max_tokens: int = 600,
    temperature: float = 0.2,
    llm_timeout: int | None = 60,
    grounding: str = "strict",
    router_backend: str = "rules",
    routing_log: str | None = None,
    routing_log_include_rationale: bool = False,
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
        router_backend=router_backend,
        llm_model=llm_model,
        llm_timeout=llm_timeout,
    )
    if routing_log:
        try:
            append_routing_event(
                {
                    "timestamp": None,  # filled in helper
                    "question": question,
                    "router_backend": router_backend,
                    "selected_backend_used": info.get("router_backend"),
                    "intent": info.get("intent"),
                    "allowed_tags": info.get("allowed_tags"),
                    "confidence": info.get("intent_confidence"),
                    "rule": info.get("intent_rule"),
                    "rationale": info.get("router_rationale")
                    if (routing_log_include_rationale or debug)
                    else None,
                    "shortlisted_docs": info.get("shortlisted_docs"),
                    "top_doc_score": info.get("top_doc_score"),
                    "top2_doc_score": info.get("top2_doc_score"),
                },
                routing_log,
            )
        except Exception as exc:  # noqa: BLE001
            if debug:
                logger.warning("routing_log_append_failed: %s", exc)
    info["backend"] = getattr(embed_model, "backend", "unknown")
    info["model_name"] = getattr(embed_model, "model_name", None) or index_meta.get("embedding", {}).get(
        "model_name"
    )
    if debug and info.get("top_score", 0.0) > 1.0 + 1e-6:
        logger.warning("Top score %.4f exceeds cosine bound; check normalization.", info.get("top_score", 0.0))

    allowed_contact_docs = set(info.get("contact_allowed_docs") or [])
    selected_hits = _select_evidence(
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
    hits_for_answer = selected_hits if selected_hits else hits

    context = "\n\n".join(
        f"[{chunk['metadata'].get('chunk_id')}] { truncate_chunk(chunk['text'], debug=debug) }"
        for chunk, _ in hits_for_answer
    )
    answer_text = None
    llm_backend_used = False
    excerpt_count = len(hits_for_answer)
    grounding_mode = (grounding or "strict").lower()
    if use_llm:
        excerpts = [
            (
                chunk["metadata"].get("chunk_id"),
                truncate_chunk(chunk["text"], debug=debug),
            )
            for chunk, _ in hits_for_answer
        ]
        raw_answer = _llm_answer(
            question,
            context,
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_timeout=llm_timeout,
            debug=debug,
            intent=info.get("intent"),
            grounding=grounding_mode,
            excerpts=excerpts,
        )
        valid = False
        used_ids: set[int] = set()
        reason = ""
        if raw_answer:
            valid, used_ids, reason = validate_llm_grounding(raw_answer, excerpt_count, grounding_mode)
        if grounding_mode == "strict":
            if raw_answer and not valid:
                if debug:
                    logger.warning("LLM grounding failed (%s), retrying with stricter instruction", reason or "unknown")
                raw_answer = _llm_answer(
                    question,
                    context,
                    model=llm_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    llm_timeout=llm_timeout,
                    debug=debug,
                    intent=info.get("intent"),
                    grounding=grounding_mode,
                    excerpts=excerpts,
                    stricter=True,
                )
                if raw_answer:
                    valid, used_ids, reason = validate_llm_grounding(raw_answer, excerpt_count, grounding_mode)
            if raw_answer and valid:
                answer_text = raw_answer
                llm_backend_used = True
            elif debug:
                logger.warning("llm_grounding_failed_fallback_to_no_llm=true")
        elif grounding_mode == "loose":
            if not raw_answer:
                valid = False
            elif not valid and reason == "invalid_citation":
                retry_answer = _llm_answer(
                    question,
                    context,
                    model=llm_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    llm_timeout=llm_timeout,
                    debug=debug,
                    intent=info.get("intent"),
                    grounding=grounding_mode,
                    excerpts=excerpts,
                    stricter=True,
                )
                if retry_answer:
                    retry_valid, _, retry_reason = validate_llm_grounding(retry_answer, excerpt_count, grounding_mode)
                    if retry_valid:
                        raw_answer = retry_answer
                        valid = True
                    else:
                        raw_answer = _strip_invalid_citations(retry_answer, excerpt_count)
                        valid = bool(raw_answer.strip())
                        if debug:
                            logger.warning(
                                "accepting_llm_answer_after_stripping_invalid_citations=true reason=%s",
                                retry_reason,
                            )
                else:
                    raw_answer = _strip_invalid_citations(raw_answer, excerpt_count)
                    valid = bool(raw_answer.strip())
                    if debug:
                        logger.warning("accepting_llm_answer_after_stripping_invalid_citations=true retry_failed=true")
            else:
                valid = bool(raw_answer)

            if raw_answer and valid:
                answer_text = raw_answer
                llm_backend_used = True
        else:  # grounding none
            if raw_answer:
                answer_text = raw_answer
                llm_backend_used = True

        if answer_text and debug:
            logger.info(
                "final_answer_backend=llm model=%s max_tokens=%d temp=%.2f grounding=%s",
                llm_model,
                max_tokens,
                temperature,
                grounding_mode,
            )

    top_score = info.get("top_score", 0.0)
    intent = info.get("intent")
    effective_threshold = REFUSAL_SCORE_THRESHOLD
    if intent in {"opening_hours", "delivery", "apostille"}:
        effective_threshold = 0.05
    if not answer_text and info.get("contact_strict"):
        answer_text = _format_contact_card(question, hits_for_answer, allowed_contact_docs)
    if not answer_text and info.get("intent") == "delivery":
        answer_text = _format_delivery_answer(hits_for_answer, question)
    if not answer_text:
        if top_score < effective_threshold:
            answer_text = "I can't find this in the provided sources. Please contact the office for details."
            confidence = False
        else:
            answer_text = _extractive_answer(question, hits_for_answer, confidence, top_k or cfg.top_k)
        if debug:
            logger.info("final_answer_backend=no_llm")

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
        "used_hits": hits_for_answer,
        "llm_backend_used": llm_backend_used,
    }
