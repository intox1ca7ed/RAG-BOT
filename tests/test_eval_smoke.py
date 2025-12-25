from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

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
    storage_root.mkdir(parents=True, exist_ok=True)

    contact_path = corpus_root / "contacts.txt"
    contact_path.write_text(
        "Contacts\n"
        "Yekaterinburg:\n"
        "Address: Malysheva street, 73A, separate entrance (VISAVISA sign)\n"
        "Opening hours: 11:00 - 18:00\n"
        "Contact phones:\n"
        "+7 (343) 271-90-09\n"
        "+7 (912) 244-37-23\n"
        "Find us on:\n"
        "WhatsApp\n"
        "Telegram\n"
        "Chelyabinsk\n"
        "Chelyabinsk, Karl Libknekhta Street, 2,\n"
        "2nd floor, office 244\n"
        "+7 (904) 309-34-81\n"
        "Mon.-Fri. from 11:00 to 19:00\n",
        encoding="utf-8",
    )

    japan_form_path = corpus_root / "form-japan.csv"
    with japan_form_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field_id", "field_label", "notes"])
        writer.writeheader()
        writer.writerow({"field_id": "jp_form", "field_label": "Japan visa form", "notes": "Basic application"})

    china_proc_path = corpus_root / "china-processing.txt"
    china_proc_path.write_text(
        "China visa processing time: standard 10-12 business days. Urgent processing 5-7 business days.",
        encoding="utf-8",
    )

    furniture_path = corpus_root / "furniture-tour.txt"
    furniture_path.write_text(
        "Furniture tour schedule for Guangzhou, curtains and bathroom furniture showcase.",
        encoding="utf-8",
    )

    taiwan_path = corpus_root / "taiwan.txt"
    taiwan_path.write_text(
        "Taiwan visa info unrelated to China processing.",
        encoding="utf-8",
    )

    manifest_records = [
        {
            "doc_id": "contacts_doc",
            "title": "Contacts",
            "local_path": str(contact_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "contact,locations,hours",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "none",
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
            "country": "japan",
        },
        {
            "doc_id": "china_processing_doc",
            "title": "China Visa Processing",
            "local_path": str(china_proc_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "china,visa,processing",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "china",
        },
        {
            "doc_id": "furniture_doc",
            "title": "Furniture Tour",
            "local_path": str(furniture_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "china,furniture,tour",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "china",
        },
        {
            "doc_id": "taiwan_doc",
            "title": "Taiwan Visa",
            "local_path": str(taiwan_path),
            "source_url": "",
            "doc_type": "txt",
            "tags": "taiwan,visa",
            "collected_at": "2025-01-01",
            "language": "en",
            "translation": "",
            "country": "taiwan",
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
    meta = embed_model.to_metadata(vectorizer_path=vec_path)
    index = IndexStore.build(vectors, chunks, meta, meta_info={"backend_requested": "simple"})
    index.save(index_dir)
    return cfg


def test_contact_query_routes_to_contacts(tmp_path: Path) -> None:
    cfg = _build_index(tmp_path)
    res = answer_question(
        "How can I find your office?",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    assert res["hits"][0][0]["metadata"]["doc_id"] == "contacts_doc"
    text_lower = res["answer_text"].lower()
    assert "malysheva" in text_lower or "karl libknekhta" in text_lower
    assert "11:00" in text_lower
    assert "curtains" not in text_lower and "bathroom furniture" not in text_lower and "furniture" not in text_lower
    assert res["retrieval_info"]["top_score"] <= 1.000001


def test_japan_form_is_retrieved(tmp_path: Path) -> None:
    cfg = _build_index(tmp_path)
    res = answer_question(
        "Japan visa form",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    assert any("Japan Visa Form" in src for src in res["sources"])
    assert res["retrieval_info"]["top_score"] <= 1.000001


def test_china_processing_time_filters_traps(tmp_path: Path) -> None:
    cfg = _build_index(tmp_path)
    res = answer_question(
        "How long does it take to make visa to china?",
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )
    text_lower = res["answer_text"].lower()
    assert "10-12" in text_lower or "5-7" in text_lower
    assert "taiwan" not in text_lower
    assert "furniture" not in text_lower
    assert res["hits"][0][0]["metadata"]["doc_id"] == "china_processing_doc"
    assert res["retrieval_info"]["top_score"] <= 1.000001


def test_eval_answers_baseline_smoke(tmp_path: Path) -> None:
    report_json = tmp_path / "eval_report.json"
    report_txt = tmp_path / "eval_report.txt"
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "eval_answers.py"),
        "--track",
        "baseline",
        "--limit",
        "3",
        "--report_json",
        str(report_json),
        "--report_txt",
        str(report_txt),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert report_json.exists()
    assert report_txt.exists()
