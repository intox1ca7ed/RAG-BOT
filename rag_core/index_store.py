from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .embeddings import EmbeddingMetadata, EmbeddingModel

logger = logging.getLogger(__name__)


def _try_import_faiss():
    try:
        import faiss                

        return faiss
    except Exception:
        return None


def _get_sklearn_version() -> str | None:
    try:
        import sklearn                

        return sklearn.__version__
    except Exception:
        return None


class SklearnVersionMismatch(Exception):
    def __init__(self, saved_version: str | None, current_version: str | None):
        msg = (
            f"Saved sklearn version {saved_version} differs from current {current_version}. "
            "Rebuild index with matching version or use --rebuild_on_mismatch."
        )
        super().__init__(msg)
        self.saved_version = saved_version
        self.current_version = current_version


class IndexStore:
    def __init__(
        self,
        vectors: np.ndarray,
        chunks: List[dict],
        backend: str,
        embedding_meta: EmbeddingMetadata,
        faiss_index: object | None = None,
        meta_info: dict | None = None,
    ):
        self.vectors = vectors.astype(np.float32)
        self.chunks = chunks
        self.backend = backend
        self.embedding_meta = embedding_meta
        self.faiss_index = faiss_index
        self.meta_info = meta_info or {}

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        chunks: List[dict],
        embedding_meta: EmbeddingMetadata,
        meta_info: dict | None = None,
    ) -> "IndexStore":
        faiss = _try_import_faiss()
        backend = "numpy"
        faiss_index = None
        vectors = cls._l2_normalize(embeddings)
        if faiss is not None:
            try:
                dim = vectors.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(vectors)
                faiss_index = index
                backend = "faiss"
                logger.info("Using FAISS index backend.")
            except Exception as exc:
                logger.warning("FAISS unavailable, falling back to numpy: %s", exc)
        return cls(vectors, chunks, backend, embedding_meta, faiss_index, meta_info)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        vectors_path = index_dir / "vectors.npy"
        embeddings_path = index_dir / "embeddings.npy"
        meta_path = index_dir / "metadata.json"
        chunks_path = index_dir / "chunks_meta.json"
        np.save(vectors_path, self.vectors)
        np.save(embeddings_path, self.vectors)
        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        if self.backend == "faiss" and self.faiss_index is not None:
            faiss = _try_import_faiss()
            if faiss:
                faiss.write_index(self.faiss_index, str(index_dir / "faiss.index"))
        meta = {
            "backend": self.backend,
            "embedding": self.embedding_meta.__dict__,
            "chunk_count": len(self.chunks),
            "dim": int(self.vectors.shape[1]) if self.vectors.size else 0,
            "sklearn_version": _get_sklearn_version() if self.embedding_meta.backend == "tfidf" else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(self.meta_info or {}),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, index_dir: Path, rebuild_on_mismatch: bool = False) -> Tuple["IndexStore", EmbeddingModel, dict]:
        vectors_path = index_dir / "vectors.npy"
        meta_path = index_dir / "metadata.json"
        legacy_meta_path = index_dir / "index_meta.json"
        chunks_path = index_dir / "chunks_meta.json"

        if not (vectors_path.exists() and (meta_path.exists() or legacy_meta_path.exists()) and chunks_path.exists()):
            raise FileNotFoundError("Index files not found in storage/index.")

        if not meta_path.exists() and legacy_meta_path.exists():
            meta_path = legacy_meta_path

        vectors = cls._l2_normalize(np.load(vectors_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        embedding_meta = EmbeddingMetadata(**meta["embedding"])

        saved_sklearn = meta.get("sklearn_version")
        current_sklearn = _get_sklearn_version()
        if embedding_meta.backend == "tfidf" and saved_sklearn and current_sklearn and saved_sklearn != current_sklearn:
            if rebuild_on_mismatch:
                logger.warning(
                    "sklearn version mismatch detected (saved=%s current=%s); rebuild requested.",
                    saved_sklearn,
                    current_sklearn,
                )
            raise SklearnVersionMismatch(saved_sklearn, current_sklearn)

        embedding_model = EmbeddingModel.load_from_metadata(embedding_meta, index_dir)

        faiss_index = None
        if meta.get("backend") == "faiss":
            faiss = _try_import_faiss()
            if faiss:
                fi = index_dir / "faiss.index"
                if fi.exists():
                    faiss_index = faiss.read_index(str(fi))
                else:
                    logger.warning("FAISS index metadata present but file missing.")
            else:
                logger.warning("FAISS backend requested but faiss not installed.")

        store = cls(
            vectors=vectors,
            chunks=chunks,
            backend=meta.get("backend", "numpy"),
            embedding_meta=embedding_meta,
            faiss_index=faiss_index,
            meta_info=meta,
        )
        return store, embedding_model, meta

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[dict, float]]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if self.backend == "faiss" and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vector.astype(np.float32), top_k)
            hits = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                hits.append((self.chunks[idx], float(score)))
            return hits

                                 
        vectors = self.vectors
        denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector, axis=1)
        denom[denom == 0] = 1.0
        sims = (vectors @ query_vector.T).reshape(-1) / denom
        top_indices = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_indices if sims[i] > 0]

    def search_filtered(
        self,
        query_vector: np.ndarray,
        top_k: int,
        candidate_indices: List[int],
    ) -> List[Tuple[dict, float]]:
        if not candidate_indices:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        vectors = self.vectors[candidate_indices]
        denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector, axis=1)
        denom[denom == 0] = 1.0
        sims = (vectors @ query_vector.T).reshape(-1) / denom
        top_indices = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score <= 0:
                continue
            global_idx = candidate_indices[idx]
            results.append((self.chunks[global_idx], score))
        return results
