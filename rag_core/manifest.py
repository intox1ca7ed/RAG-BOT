from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .config import Config, DEFAULT_CONFIG


REQUIRED_FIELDS = [
    "doc_id",
    "title",
    "local_path",
    "source_url",
    "doc_type",
    "tags",
    "collected_at",
]

OPTIONAL_FIELDS = ["language", "translation"]


def _normalize_record(raw: dict) -> dict:
    record = {key: raw.get(key, "") for key in REQUIRED_FIELDS + OPTIONAL_FIELDS}
                                                         
    for key, value in raw.items():
        if key not in record:
            record[key] = value
    return record


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [_normalize_record(row) for row in reader]


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        return [_normalize_record(item) for item in data]


def load_manifest(config: Config | None = None) -> list[dict]:
    cfg = config or DEFAULT_CONFIG
    if cfg.manifest_csv.exists():
        records = load_csv(cfg.manifest_csv)
    elif cfg.manifest_json.exists():
        records = load_json(cfg.manifest_json)
    else:
        raise FileNotFoundError("No manifest.csv or manifest.json found.")
    return _filter_empty_doc_ids(records)


def _filter_empty_doc_ids(records: Iterable[dict]) -> list[dict]:
    return [r for r in records if r.get("doc_id")]
