from __future__ import annotations

import csv
from pathlib import Path

from rag_core.chunking import chunk_corpus
from rag_core.config import Config
from rag_core.embeddings import EmbeddingModel
from rag_core.index_store import IndexStore
from rag_core.loaders import load_document
from rag_core.rag import answer_question


def _build_index(tmp_path: Path) -> Config:
    storage_root = tmp_path / "storage"
    corpus_root = tmp_path
    cfg = Config(storage_root=storage_root, corpus_root=corpus_root)

    duties_path = corpus_root / "furniture_duties.csv"
    with duties_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field_id", "field_label", "notes"])
        writer.writeheader()
        writer.writerow(
            {"field_id": "duty1", "field_label": "Furniture duty", "notes": "Rate: $4.50 per 1 kg."}
        )

    tour_cost_path = corpus_root / "furniture_tour.txt"
    tour_cost_path.write_text("Furniture tour cost: 600 USD per person, includes transport.", encoding="utf-8")

    manifest_records = [
        {
            "doc_id": "furniture_tour_cost_doc",
            "title": "Furniture Tour Cost",
            "local_path": str(tour_cost_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "furniture,tour,cost",
            "topic": "furniture_tour",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "china",
        },
        {
            "doc_id": "furniture_tour_cost_duties_csv",
            "title": "Furniture Tour Cost Duties",
            "local_path": str(duties_path),
            "source_url": "",
            "doc_type": "csv",
            "tags": "furniture,duties,customs",
            "topic": "customs_duties",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "china",
        },
    ]

    docs = [load_document(rec, cfg) for rec in manifest_records]
    docs = [d for d in docs if d]
    chunks = chunk_corpus(docs, chunk_size=400, chunk_overlap=50)

    embed_model = EmbeddingModel(prefer="tfidf")
    vectors = embed_model.fit([c["text"] for c in chunks])
    index_dir = cfg.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    vec_path = index_dir / "vectorizer.pkl"
    if embed_model.backend == "tfidf":
        embed_model.save_vectorizer(vec_path)
    meta = embed_model.to_metadata(vectorizer_path=vec_path if embed_model.backend == "tfidf" else None)
    index = IndexStore.build(vectors, chunks, meta, meta_info={"backend_requested": embed_model.backend})
    index.save(index_dir)
    return cfg


def test_furniture_cost_excludes_duties(tmp_path: Path) -> None:
    cfg = _build_index(tmp_path)
    res = answer_question(
        "How much does the furniture tour cost?",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    doc_ids = [c.get("metadata", {}).get("doc_id", "") for c, _ in res["hits"]]
    assert "furniture_tour_cost_duties_csv" not in doc_ids
    texts = [c.get("text", "").lower() for c, _ in res["hits"]]
    assert not any("rate:" in t and "per 1 kg" in t for t in texts)
