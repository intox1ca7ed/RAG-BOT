from __future__ import annotations

from pathlib import Path

from rag_core.chunking import chunk_corpus
from rag_core.config import Config
from rag_core.embeddings import EmbeddingModel
from rag_core.index_store import IndexStore
from rag_core.loaders import load_document
from rag_core.rag import answer_question


def _build_ops_index(tmp_path: Path) -> Config:
    storage_root = tmp_path / "storage"
    corpus_root = tmp_path
    cfg = Config(storage_root=storage_root, corpus_root=corpus_root)

    contacts_path = corpus_root / "contacts.txt"
    contacts_path.write_text(
        "Contacts\nOpening hours: Mon-Fri 11:00-19:00. Saturdays closed. Address: Yekaterinburg, 123 Main St.",
        encoding="utf-8",
    )

    main_page_path = corpus_root / "main-page.txt"
    main_page_path.write_text(
        "Production time 10-12 business days, urgent 5-7 business days.\n"
        "Apostille - Legalization of Documents for China\n"
        "19,000 rubles for 8 working days\n"
        "30,000 rubles for 3 working days\n"
        "The processing time starts after the original documents are received at the Moscow office.\n"
        "You can send the documents by express mail or flying envelope of S7 or Ural Airlines.\n",
        encoding="utf-8",
    )

    manifest_records = [
        {
            "doc_id": "contacts_doc",
            "title": "Contacts",
            "local_path": str(contacts_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "contact,hours",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "none",
        },
        {
            "doc_id": "main_page_doc",
            "title": "Main Page",
            "local_path": str(main_page_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "main_page,overview,china,visa,processing,apostille,legalization,delivery",
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


def test_opening_hours_query(tmp_path: Path) -> None:
    cfg = _build_ops_index(tmp_path)
    res = answer_question(
        "are you open on saturdays?",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    text_lower = res["answer_text"].lower()
    assert "saturday" in text_lower or "11:00" in text_lower
    assert any("contacts" in src for src in res["sources"]) or any("main page" in src.lower() for src in res["sources"])
    assert res["retrieval_info"]["intent"] == "opening_hours"


def test_delivery_query(tmp_path: Path) -> None:
    cfg = _build_ops_index(tmp_path)
    res = answer_question(
        "I live in another city. How can I send my passport to you?",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    text_lower = res["answer_text"].lower()
    assert "express" in text_lower or "flying envelope" in text_lower or "s7" in text_lower
    assert any("main page" in src.lower() for src in res["sources"])
    assert "notarized" not in text_lower and "list of documents" not in text_lower


def test_apostille_query(tmp_path: Path) -> None:
    cfg = _build_ops_index(tmp_path)
    res = answer_question(
        "apostille urgently: cost + when ready?",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    text_lower = res["answer_text"].lower()
    assert "19,000" in text_lower or "19000" in text_lower
    assert "30,000" in text_lower or "30000" in text_lower
    assert any("main page" in src.lower() for src in res["sources"])


def test_main_page_not_excluded_processing(tmp_path: Path) -> None:
    cfg = _build_ops_index(tmp_path)
    res = answer_question(
        "How long does it take to make visa to china?",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    assert any("main page" in src.lower() for src in res["sources"])
