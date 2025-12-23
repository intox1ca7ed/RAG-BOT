from __future__ import annotations

from rag_core.rag import _strip_invalid_citations, validate_llm_grounding


def test_validate_strict_requires_citation() -> None:
    ok, ids, reason = validate_llm_grounding("Answer without cites", 2, "strict")
    assert not ok
    assert ids == set()
    assert reason == "no_citations"


def test_validate_strict_accepts_bracket_and_sources_line() -> None:
    ok1, ids1, _ = validate_llm_grounding("Look here [S1]", 2, "strict")
    ok2, ids2, _ = validate_llm_grounding("Answer...\nSources used: S2", 3, "strict")
    assert ok1 and ids1 == {1}
    assert ok2 and ids2 == {2}


def test_validate_loose_allows_missing_citations() -> None:
    ok, ids, reason = validate_llm_grounding("No citations provided", 2, "loose")
    assert ok
    assert ids == set()
    assert reason in {"ok", "no_citations"}


def test_validate_loose_strips_invalid_ids() -> None:
    ok, _, reason = validate_llm_grounding("Sources used: S99", 2, "loose")
    assert not ok
    assert reason == "invalid_citation"
    cleaned = _strip_invalid_citations("Sources used: S99", 2)
    ok_clean, ids_clean, _ = validate_llm_grounding(cleaned, 2, "loose")
    assert ok_clean
    assert ids_clean == set()


def test_validate_none_always_passes_non_empty() -> None:
    ok, ids, reason = validate_llm_grounding("Any text", 0, "none")
    assert ok
    assert ids == set()
    assert reason == "mode_none"


def test_validate_ranges_and_spacing() -> None:
    text = "Sources used: S1-2,  S3 (see [S2])"
    ok, ids, _ = validate_llm_grounding(text, 3, "strict")
    assert ok
    assert ids == {1, 2, 3}


def test_strict_flags_invalid_id() -> None:
    ok, ids, reason = validate_llm_grounding("Sources used: S4", 2, "strict")
    assert not ok
    assert ids == set()
    assert reason == "invalid_citation"
