from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

from .config import Config, DEFAULT_CONFIG, REPO_ROOT

logger = logging.getLogger(__name__)


SUPPORTED_DOC_TYPES = {"txt", "md", "csv"}


def _resolve_path(local_path: str, config: Config) -> Path:
    path = Path(local_path)
    if path.is_absolute():
        return path

    candidate = path
                                                                                 
    if candidate.parts and candidate.parts[0] == config.corpus_root.name:
        candidate = Path(*candidate.parts[1:])
    resolved = (config.corpus_root / candidate).resolve()
    return resolved


def _should_skip(record: dict, file_path: Path) -> bool:
    doc_type = (record.get("doc_type") or file_path.suffix.lstrip(".")).lower()
    if "pdf-raw" in file_path.as_posix().lower():
        return True
    if doc_type == "pdf":
        return True
    if doc_type not in SUPPORTED_DOC_TYPES:
        logger.warning("Skipping unsupported doc_type=%s for %s", doc_type, file_path)
        return True
    return False


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _warn_replacements(text: str, path: Path) -> None:
    repl_count = text.count("\ufffd")
    if repl_count == 0:
        return
                                                                                               
    ratio = repl_count / max(len(text), 1)
    if repl_count >= 5 or ratio > 0.001:
        logger.warning(
            "Replacement characters detected in %s (count=%d, ratio=%.5f). Consider re-saving as UTF-8.",
            path,
            repl_count,
            ratio,
        )


def _load_txt(path: Path) -> tuple[str, dict]:
    content = path.read_text(encoding="utf-8", errors="replace")
    _warn_replacements(content, path)
    return _normalize_text(content), {}


def _load_md(path: Path) -> tuple[str, dict]:
    return _load_txt(path)


def _format_csv_line(row: dict, special: bool) -> str:
    if special:
        return (
            f"field_label: {row.get('field_label', '').strip()}; "
            f"notes: {row.get('notes', '').strip()}; "
            f"(field_id: {row.get('field_id', '').strip()})"
        )
    parts = [f"{k}: {v}".strip() for k, v in row.items() if v not in (None, "")]
    return "; ".join(p for p in parts if p)


def _load_csv(path: Path) -> tuple[str, dict, list[dict]]:
    """
    Returns normalized content (joined rows), metadata, and per-row data for chunking.
    """
    row_lines = []
    csv_metadata = {"csv_row_count": 0}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.lower() for fn in (reader.fieldnames or [])]
        special = {"field_id", "field_label", "notes"}.issubset(set(fieldnames))
        for idx, row in enumerate(reader):
            line = _format_csv_line(row, special)
            row_lines.append({"row_index": idx, "text": line})
        csv_metadata["csv_row_count"] = len(row_lines)
    combined = "\n".join(r["text"] for r in row_lines)
    _warn_replacements(combined, path)
    return _normalize_text(combined), csv_metadata, row_lines


def load_document(record: dict, config: Config | None = None) -> dict | None:
    cfg = config or DEFAULT_CONFIG
    local_path_str = record.get("local_path", "")
    file_path = _resolve_path(local_path_str, cfg)
    if _should_skip(record, file_path):
        return None
    if not file_path.exists():
        logger.warning("File missing for doc_id=%s path=%s", record.get("doc_id"), file_path)
        return None

    doc_type = (record.get("doc_type") or file_path.suffix.lstrip(".")).lower()
    csv_rows_data: list[dict] | None = None
    if doc_type == "txt":
        content, extra_meta = _load_txt(file_path)
    elif doc_type == "md":
        content, extra_meta = _load_md(file_path)
    elif doc_type == "csv":
        content, extra_meta, csv_rows_data = _load_csv(file_path)
    else:
        return None

    try:
        rel_path = file_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_path = file_path.as_posix()

    metadata = {
        "doc_id": record.get("doc_id", ""),
        "title": record.get("title", ""),
        "source_url": record.get("source_url", ""),
        "local_path": rel_path,
        "doc_type": doc_type,
        "tags": record.get("tags", ""),
        "collected_at": record.get("collected_at", ""),
        "language": record.get("language", ""),
        "translation": record.get("translation", ""),
        **extra_meta,
    }
    doc = {"content": content, "metadata": metadata}
    if csv_rows_data is not None:
        doc["csv_rows"] = csv_rows_data
    return doc


def load_corpus(records: Iterable[dict], config: Config | None = None) -> list[dict]:
    docs = []
    for record in records:
        doc = load_document(record, config)
        if doc:
            docs.append(doc)
    return docs
