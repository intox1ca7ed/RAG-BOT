from __future__ import annotations

from typing import List, Tuple

import logging
import numpy as np

from .config import Config, DEFAULT_CONFIG
from .embeddings import EmbeddingModel
from .index_store import IndexStore

logger = logging.getLogger(__name__)


def _detect_allowed_tags(question: str) -> set[str]:
    q = question.lower()
    allowed = set()
    if "china" in q:
        allowed.add("china")
    if "japan" in q:
        allowed.add("japan")
    if "taiwan" in q:
        allowed.add("taiwan")
    if any(k in q for k in ["furniture", "tour", "guangzhou", "foshan"]):
        allowed.add("furniture")
    if any(k in q for k in ["legalization", "apostille", "legal"]):
        allowed.add("legalization")
    if "visa" in q:
        allowed.add("visa")
    return allowed


def _tags_set(meta_tags: str | list | None) -> set[str]:
    if meta_tags is None:
        return set()
    if isinstance(meta_tags, list):
        items = meta_tags
    else:
        items = str(meta_tags).split(",")
    return {t.strip().lower() for t in items if t and str(t).strip()}


def retrieve(
    question: str,
    index_store: IndexStore,
    embed_model: EmbeddingModel,
    top_k: int | None = None,
    config: Config | None = None,
) -> tuple[List[tuple[dict, float]], bool, dict]:
    cfg = config or DEFAULT_CONFIG
    k = top_k or cfg.top_k
    allowed_tags = _detect_allowed_tags(question)
    query_vec = embed_model.encode([question])[0]

    filtered_indices = []
    if allowed_tags:
        for idx, chunk in enumerate(index_store.chunks):
            meta_tags = _tags_set(chunk.get("metadata", {}).get("tags"))
            if meta_tags & allowed_tags:
                filtered_indices.append(idx)

    filtered_applied = False
    filter_warning = False
    if allowed_tags and len(filtered_indices) >= 30:
        hits = index_store.search_filtered(np.array(query_vec), k, filtered_indices)
        filtered_applied = True
    else:
        if allowed_tags and len(filtered_indices) < 30:
            filter_warning = True
            logger.warning(
                "Tag filtering returned %s candidates (<30); using unfiltered retrieval.",
                len(filtered_indices),
            )
        hits = index_store.search(np.array(query_vec), k)

    boost_condition = all(word in question.lower() for word in ["where", "buy", "furniture"])
    if boost_condition:
        boosted = []
        for chunk, score in hits:
            meta = chunk.get("metadata", {})
            tags = _tags_set(meta.get("tags"))
            doc_id = meta.get("doc_id", "").lower()
            if "furniture" in tags and "places" in doc_id:
                score *= 1.1
            boosted.append((chunk, score))
        hits = sorted(boosted, key=lambda x: x[1], reverse=True)

    top_score = hits[0][1] if hits else 0.0
    confidence = bool(hits) and top_score >= cfg.confidence_threshold
    info = {
        "allowed_tags": sorted(allowed_tags),
        "filtered_applied": filtered_applied,
        "filter_warning": filter_warning,
        "top_score": top_score,
    }
    return hits, confidence, info
