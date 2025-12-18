from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np

logger = logging.getLogger(__name__)


def _try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except Exception:  # ImportError or other runtime issues
        return None


def _try_import_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer
    except Exception:
        return None


@dataclass
class EmbeddingMetadata:
    backend: str
    model_name: str | None = None
    vectorizer_path: str | None = None


class EmbeddingModel:
    def __init__(self, prefer: str | None = None):
        self.backend = "simple"
        self.model_name: str | None = None
        self.vectorizer = None
        self._st_model = None

        SentenceTransformer = _try_import_sentence_transformers()
        if (prefer in (None, "sentence-transformers")) and SentenceTransformer:
            try:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.backend = "sentence-transformers"
                self.model_name = "all-MiniLM-L6-v2"
                logger.info("Using sentence-transformers backend.")
                return
            except Exception as exc:
                logger.warning("Failed to init sentence-transformers: %s", exc)

        TfidfVectorizer = _try_import_sklearn()
        if (prefer in (None, "tfidf", "simple")) and TfidfVectorizer:
            self.backend = "tfidf"
            self.vectorizer = TfidfVectorizer(stop_words="english")
            logger.info("Using TF-IDF backend.")
            return

        logger.info("Falling back to simple bag-of-words embeddings.")

    def fit(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if self.backend == "sentence-transformers":
            return self._encode_st(texts_list)
        if self.backend == "tfidf" and self.vectorizer is not None:
            vectors = self.vectorizer.fit_transform(texts_list).toarray()
            return self._normalize(vectors)
        # simple backend
        return self._simple_embed(texts_list)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if self.backend == "sentence-transformers":
            return self._encode_st(texts_list)
        if self.backend == "tfidf" and self.vectorizer is not None:
            vectors = self.vectorizer.transform(texts_list).toarray()
            return self._normalize(vectors)
        return self._simple_embed(texts_list)

    def _encode_st(self, texts: List[str]) -> np.ndarray:
        embeddings = self._st_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings

    def _simple_embed(self, texts: List[str]) -> np.ndarray:
        vocab: dict[str, int] = {}
        rows = []
        for text in texts:
            tokens = text.lower().split()
            counts: dict[int, float] = {}
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
                idx = vocab[tok]
                counts[idx] = counts.get(idx, 0.0) + 1.0
            row = np.zeros(len(vocab), dtype=np.float32)
            for idx, val in counts.items():
                if idx >= len(row):
                    # extend row if vocab grew mid-document
                    row = np.pad(row, (0, idx - len(row) + 1))
                row[idx] = val
            rows.append(row)
        max_dim = max((r.shape[0] for r in rows), default=0)
        padded = np.zeros((len(rows), max_dim), dtype=np.float32)
        for i, row in enumerate(rows):
            padded[i, : row.shape[0]] = row
        return self._normalize(padded)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def to_metadata(self, vectorizer_path: Path | None = None) -> EmbeddingMetadata:
        return EmbeddingMetadata(
            backend=self.backend,
            model_name=self.model_name,
            vectorizer_path=str(vectorizer_path) if vectorizer_path else None,
        )

    def save_vectorizer(self, path: Path) -> None:
        if self.backend == "tfidf" and self.vectorizer is not None:
            with path.open("wb") as f:
                pickle.dump(self.vectorizer, f)

    @classmethod
    def load_from_metadata(cls, meta: EmbeddingMetadata, index_dir: Path) -> "EmbeddingModel":
        model = cls(prefer=meta.backend)
        model.backend = meta.backend
        model.model_name = meta.model_name
        if meta.backend == "tfidf" and meta.vectorizer_path:
            vec_path = Path(meta.vectorizer_path)
            if not vec_path.is_absolute():
                vec_path = index_dir / vec_path
            if vec_path.exists():
                with vec_path.open("rb") as f:
                    model.vectorizer = pickle.load(f)
            else:
                logger.warning("Vectorizer file missing at %s", vec_path)
        return model
