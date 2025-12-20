from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rag_core.chunking import chunk_corpus
from rag_core.config import Config
from rag_core.embeddings import EmbeddingModel
from rag_core.index_store import IndexStore
from rag_core.loaders import load_document
from rag_core.rag import answer_question


def _has_cross_encoder() -> bool:
    try:
        spec = importlib.util.find_spec("sentence_transformers.cross_encoder")
        return spec is not None
    except ModuleNotFoundError:
        return False


def _build_index(tmp_path: Path) -> Config:
    storage_root = tmp_path / "storage"
    corpus_root = tmp_path
    cfg = Config(storage_root=storage_root, corpus_root=corpus_root)

    processing_path = corpus_root / "china-processing.txt"
    processing_path.write_text(
        "Processing time: standard 10-12 business days. Urgent processing 5-7 business days.",
        encoding="utf-8",
    )

    transit_path = corpus_root / "china-transit.txt"
    transit_path.write_text(
        "144-hour visa-free transit rules for China airports.",
        encoding="utf-8",
    )

    manifest_records = [
        {
            "doc_id": "china_processing_doc",
            "title": "China Visa Processing",
            "local_path": str(processing_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "china,visa,processing",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "china",
        },
        {
            "doc_id": "china_transit_doc",
            "title": "China Transit",
            "local_path": str(transit_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "china,transit,visa",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "china",
        },
    ]

    docs = [load_document(rec, cfg) for rec in manifest_records]
    docs = [d for d in docs if d]
    chunks = chunk_corpus(docs, chunk_size=400, chunk_overlap=50)

    embed_model = EmbeddingModel(prefer="simple")
    vectors = embed_model.fit([c["text"] for c in chunks])
    index_dir = cfg.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    meta = embed_model.to_metadata()
    index = IndexStore.build(vectors, chunks, meta, meta_info={"backend_requested": "simple"})
    index.save(index_dir)
    return cfg


def test_rerank_prefers_processing_over_transit(tmp_path: Path) -> None:
    if not _has_cross_encoder():
        pytest.skip("CrossEncoder not available")
    cfg = _build_index(tmp_path)
    res = answer_question(
        "How long does it take to make visa to China?",
        top_k=3,
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
        rerank=True,
        rerank_mode="on",
        rerank_top_n=5,
    )
    assert res["retrieval_info"].get("rerank_used")
    assert res["hits"][0][0]["metadata"]["doc_id"] == "china_processing_doc"
    assert "144" not in res["answer_text"]
