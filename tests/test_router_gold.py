from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_core.config import Config
from rag_core.index_store import IndexStore
from rag_core.retrieval import _is_rules_confident, _rules_router, retrieve


def _load_gold() -> list[dict]:
    path = Path(__file__).parent / "routing_gold.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_rules_router_gold_set() -> None:
    gold = _load_gold()
    cfg = Config()
    try:
        index_store, embed_model, _ = IndexStore.load(cfg.index_dir, rebuild_on_mismatch=False)
    except RuntimeError as exc:
        if "sentence-transformers" in str(exc):
            pytest.skip("sentence-transformers backend not available for router gold test")
        raise

    matches = 0
    total = 0
    failures: list[str] = []
    known_failures: set[str] = set()  # populate if needed to land incremental fixes

    for sample in gold:
        q = sample["question"]
        if q in known_failures:
            continue
        expected_intent = sample["expected_intent"]
        hits, _conf, info = retrieve(
            q,
            index_store,
            embed_model,
            top_k=cfg.top_k,
            config=cfg,
            debug=False,
            router_backend="rules",
        )
        actual_intent = info.get("intent")
        total += 1
        if actual_intent == expected_intent:
            matches += 1
        else:
            failures.append(f"Q: {q} expected {expected_intent} got {actual_intent}")

    accuracy = matches / max(total, 1)
    assert accuracy >= 0.8, f"Rules router accuracy {accuracy:.2f} below threshold. Failures: {failures}"


def test_rules_confidence_triggers_llm_for_ambiguous() -> None:
    cfg = Config()
    ambiguous_questions = [
        "Can you help me with my situation?",
        "I need some information please.",
        "What can you do for me?",
        "I am not sure what I need, can you help?",
    ]
    for q in ambiguous_questions:
        router = _rules_router(q)
        assert not _is_rules_confident(router, q, debug=False)
