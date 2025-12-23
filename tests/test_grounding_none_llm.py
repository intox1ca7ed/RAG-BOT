from __future__ import annotations

from rag_core import rag


def test_grounding_none_uses_llm(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    class DummyEmbed:
        backend = "local"
        model_name = "dummy"

    dummy_chunk = {
        "metadata": {"chunk_id": "c1", "doc_id": "doc1", "title": "Doc 1"},
        "text": "Sample text about costs.",
    }

    def fake_load(index_dir, rebuild_on_mismatch=False):
        return None, DummyEmbed(), {"embedding": {"model_name": "dummy"}, "backend": "local"}

    def fake_retrieve(*args, **kwargs):
        hits = [(dummy_chunk, 0.9)]
        info = {
            "intent": "none",
            "intent_confidence": 1.0,
            "intent_rule": "test",
            "allowed_tags": [],
            "filtered_applied": False,
            "fallback_used": False,
            "fallback_reasons": [],
            "contact_allowed_docs": [],
            "contact_strict": False,
            "processing_country": None,
            "rerank_used": False,
            "rerank_model": None,
            "rerank_device": None,
            "rerank_top_n": None,
            "rerank_top_k": None,
            "rerank_changed": 0,
            "shortlisted_docs": ["doc1"],
            "top_score": 0.9,
            "prefer_china_default": False,
            "rerank_table": [],
            "delivery_debug": [],
        }
        return hits, True, info

    monkeypatch.setattr(rag.IndexStore, "load", staticmethod(fake_load))
    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "_llm_answer", lambda *args, **kwargs: "LLM generated answer.")

    result = rag.answer_question(
        question="How much does the consultant cost per day?",
        use_llm=True,
        grounding="none",
        llm_model="gpt-4o-mini",
    )

    assert result["answer_text"] == "LLM generated answer."
    assert result["llm_backend_used"] is True
    assert result["retrieval_info"]["intent"] == "none"
