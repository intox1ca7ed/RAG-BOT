from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    manifest_csv: Path = REPO_ROOT / "manifest.csv"
    manifest_json: Path = REPO_ROOT / "manifest.json"
    corpus_root: Path = REPO_ROOT / "RAG-corpus"
    storage_root: Path = REPO_ROOT / "storage"
    chunk_size: int = 1400
    chunk_overlap: int = 200
    top_k: int = 6
    confidence_threshold: float = 0.25

    @property
    def chunks_path(self) -> Path:
        return self.storage_root / "chunks" / "chunks.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.storage_root / "index"

    @property
    def logs_dir(self) -> Path:
        return self.storage_root / "logs"

    def ensure_storage_dirs(self) -> None:
        for path in (self.chunks_path.parent, self.index_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = Config()
