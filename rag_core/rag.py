from __future__ import annotations

import logging
import os
import re
from typing import Iterable, List, Tuple

from .config import Config, DEFAULT_CONFIG
from .embeddings import EmbeddingModel
from .index_store import IndexStore, SklearnVersionMismatch
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retrieval import retrieve

logger = logging.getLogger(__name__)


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
        return "I couldn't find an exact rule in the sources; please use handoff."
    sentences = _select_sentences(hits[:top_k], question, max_lines=8)
    lines: List[str] = []
    if not confidence:
        lines.append("I couldn't find an exact rule in the sources; please use handoff.")
    if sentences:
        lines.extend(f"- {s}" for s in sentences)
    else:
        lines.append("No matching sentences found in context.")
    return "\n".join(lines)


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


def answer_question(
    question: str,
    top_k: int | None = None,
    use_llm: bool = True,
    show_context: bool = False,
    config: Config | None = None,
    rebuild_on_mismatch: bool = False,
) -> dict:
    cfg = config or DEFAULT_CONFIG
    index_store, embed_model = IndexStore.load(cfg.index_dir, rebuild_on_mismatch=rebuild_on_mismatch)
    hits, confidence, info = retrieve(question, index_store, embed_model, top_k=top_k, config=cfg)

    context = "\n\n".join(
        f"[{chunk['metadata'].get('chunk_id')}] {chunk['text']}" for chunk, _ in hits
    )
    answer_text = None
    if use_llm:
        answer_text = _llm_answer(question, context)

    if not answer_text:
        answer_text = _extractive_answer(question, hits, confidence, top_k or cfg.top_k)

    sources = _format_sources(hits)
    context_lines = []
    if show_context and hits:
        for chunk, score in hits:
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
    }
