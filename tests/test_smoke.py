from __future__ import annotations

import csv
from pathlib import Path

from rag_core.chunking import chunk_corpus
from rag_core.config import Config
from rag_core.embeddings import EmbeddingModel
from rag_core.index_store import IndexStore
from rag_core.loaders import load_document
from rag_core.retrieval import retrieve


def test_smoke(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    cfg = Config(storage_root=storage_root)

    # Create sample txt
    txt_path = tmp_path / "doc1.txt"
    txt_path.write_text("This is a sample visa information text for testing retrieval.", encoding="utf-8")

    # Create sample csv with special columns
    csv_path = tmp_path / "doc2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field_id", "field_label", "notes"])
        writer.writeheader()
        writer.writerow({"field_id": "F1", "field_label": "Passport number", "notes": "Required"})

    manifest_records = [
        {
            "doc_id": "doc1",
            "title": "Doc One",
            "local_path": str(txt_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "visa",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
        },
        {
            "doc_id": "doc2",
            "title": "Doc Two",
            "local_path": str(csv_path),
            "source_url": "",
            "doc_type": "csv",
            "tags": "form",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
        },
    ]

    docs = [load_document(rec, cfg) for rec in manifest_records]
    docs = [d for d in docs if d]
    chunks = chunk_corpus(docs, chunk_size=500, chunk_overlap=50)

    embed_model = EmbeddingModel(prefer="simple")
    vectors = embed_model.fit([c["text"] for c in chunks])
    index = IndexStore.build(vectors, chunks, embed_model.to_metadata())

    hits, confidence, info = retrieve("visa passport", index, embed_model, top_k=2, config=cfg)
    assert hits, "Expected retrieval hits from smoke test corpus"
    assert isinstance(confidence, bool)
    assert "allowed_tags" in info
