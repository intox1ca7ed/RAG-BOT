from __future__ import annotations

import csv
import importlib.util
import pytest
from pathlib import Path

from rag_core.chunking import chunk_corpus
from rag_core.config import Config
from rag_core.embeddings import EmbeddingModel
from rag_core.index_store import IndexStore
from rag_core.loaders import load_document
from rag_core.rag import answer_question


def _has_sentence_transformers() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(not _has_sentence_transformers(), reason="sentence-transformers not installed")
def test_embeddings_backend_contact_and_form(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    corpus_root = tmp_path
    cfg = Config(storage_root=storage_root, corpus_root=corpus_root)

    contact_path = corpus_root / "contacts.txt"
    contact_path.write_text(
        "Contacts: Office at 123 Main St. Phone +7 123 456. Hours 10-19.",
        encoding="utf-8",
    )
    japan_form_path = corpus_root / "form-japan.csv"
    with japan_form_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field_id", "field_label", "notes"])
        writer.writeheader()
        writer.writerow({"field_id": "jp_form", "field_label": "Japan visa form", "notes": "Basic application"})

    manifest_records = [
        {
            "doc_id": "contacts_doc",
            "title": "Contacts",
            "local_path": str(contact_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "contact,locations",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
        },
        {
            "doc_id": "japan_form_doc",
            "title": "Japan Visa Form",
            "local_path": str(japan_form_path),
            "source_url": "",
            "doc_type": "csv",
            "tags": "form,japan,visa",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
        },
    ]

    docs = [load_document(rec, cfg) for rec in manifest_records]
    docs = [d for d in docs if d]
    chunks = chunk_corpus(docs, chunk_size=400, chunk_overlap=50)

    embed_model = EmbeddingModel(prefer="sentence-transformers", model_name="all-MiniLM-L6-v2")
    vectors = embed_model.fit([c["text"] for c in chunks])
    index_dir = cfg.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    meta = embed_model.to_metadata()
    index = IndexStore.build(vectors, chunks, meta, meta_info={"backend_requested": "local"})
    index.save(index_dir)

    res_contact = answer_question(
        "How can I find your office?",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    assert res_contact["retrieval_info"].get("backend") == "sentence-transformers"
    assert any("Contacts" in src for src in res_contact["sources"])

    res_form = answer_question(
        "Japan visa form",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    assert any("Japan Visa Form" in src for src in res_form["sources"])
