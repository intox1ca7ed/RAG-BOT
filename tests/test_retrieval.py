from __future__ import annotations

import csv
from pathlib import Path

from rag_core.chunking import chunk_corpus
from rag_core.config import Config
from rag_core.embeddings import EmbeddingModel
from rag_core.index_store import IndexStore
from rag_core.loaders import load_document
from rag_core.rag import answer_question


def _build_test_index(tmp_path: Path) -> Config:
    storage_root = tmp_path / "storage"
    corpus_root = tmp_path
    cfg = Config(storage_root=storage_root, corpus_root=corpus_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    contact_path = corpus_root / "contacts.txt"
    contact_path.write_text(
        "Contacts\nOffice: 123 Main St, Yekaterinburg. Phone +7 123 456 7890. "
        "Hours: Mon-Fri 11:00-19:00. Email: office@example.com",
        encoding="utf-8",
    )

    china_path = corpus_root / "visas-china.txt"
    china_path.write_text(
        "China visa requirements: passport, photo, questionnaire. Processing in Moscow office.",
        encoding="utf-8",
    )

    japan_form_path = corpus_root / "form-japan.csv"
    with japan_form_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field_id", "field_label", "notes"])
        writer.writeheader()
        writer.writerow({"field_id": "jp_f1", "field_label": "Passport copy", "notes": "Required for Japan visa"})

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
            "country": "none",
        },
        {
            "doc_id": "visa_china_doc",
            "title": "China Visa",
            "local_path": str(china_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "visa,china",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "none",
        },
        {
            "doc_id": "visa_japan_form_doc",
            "title": "Japan Visa Form",
            "local_path": str(japan_form_path),
            "source_url": "",
            "doc_type": "csv",
            "tags": "form,japan,visa",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "none",
        },
    ]

    docs = [load_document(rec, cfg) for rec in manifest_records]
    docs = [d for d in docs if d]
    chunks = chunk_corpus(docs, chunk_size=400, chunk_overlap=50)

    embed_model = EmbeddingModel(prefer="simple")
    vectors = embed_model.fit([c["text"] for c in chunks])
    index_dir = cfg.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    meta = embed_model.to_metadata(vectorizer_path=index_dir / "vectorizer.pkl")
    if embed_model.backend == "tfidf":
        embed_model.save_vectorizer(index_dir / "vectorizer.pkl")
    index = IndexStore.build(vectors, chunks, meta)
    index.save(index_dir)
    return cfg


def test_contact_intent_prefers_contacts(tmp_path: Path) -> None:
    cfg = _build_test_index(tmp_path)
    result = answer_question(
        "How can I find your office?",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    assert any("Contacts" in src for src in result["sources"])
    assert "office" in result["answer_text"].lower() or "address" in result["answer_text"].lower()


def test_japan_form_routing(tmp_path: Path) -> None:
    cfg = _build_test_index(tmp_path)
    result = answer_question(
        "Japan visa form",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    assert any("Japan Visa Form" in src for src in result["sources"])
    assert all("china" not in src.lower() for src in result["sources"][:1])


def test_china_visa_not_japan(tmp_path: Path) -> None:
    cfg = _build_test_index(tmp_path)
    result = answer_question(
        "China visa requirements",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    # First source should be China-related, not Japan
    assert "China" in result["sources"][0]
    assert "Japan" not in result["sources"][0]


def test_refusal_when_absent(tmp_path: Path) -> None:
    cfg = _build_test_index(tmp_path)
    result = answer_question(
        "What is the weather on Mars tomorrow?",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    assert "can't find" in result["answer_text"].lower()
