from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

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
    def __init__(self, prefer: str | None = None, model_name: str | None = None):
        self.backend = "simple"
        self.model_name: str | None = model_name
        self.vectorizer = None
        self._st_model = None
        self._simple_vocab: dict[str, int] | None = None

        SentenceTransformer = _try_import_sentence_transformers()
        if (prefer in (None, "sentence-transformers", "local", "st", "auto")) and SentenceTransformer:
            name = model_name or "all-MiniLM-L6-v2"
            try:
                self._st_model = SentenceTransformer(name)
                self.backend = "sentence-transformers"
                self.model_name = name
                logger.info("Using sentence-transformers backend (%s).", name)
                return
            except Exception as exc:
                logger.warning("Failed to init sentence-transformers (%s): %s", name, exc)

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
        self._simple_vocab = self._build_simple_vocab(texts_list)
        return self._simple_encode(texts_list)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if self.backend == "sentence-transformers":
            return self._encode_st(texts_list)
        if self.backend == "tfidf" and self.vectorizer is not None:
            vectors = self.vectorizer.transform(texts_list).toarray()
            return self._normalize(vectors)
        if self._simple_vocab is None:
            raise RuntimeError("Simple embeddings require a fitted vocab. Rebuild the index or call fit() first.")
        return self._simple_encode(texts_list)

    def _encode_st(self, texts: List[str]) -> np.ndarray:
        embeddings = self._st_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        return embeddings

    def _build_simple_vocab(self, texts: List[str]) -> dict[str, int]:
        vocab: dict[str, int] = {}
        for text in texts:
            tokens = text.lower().split()
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        return vocab

    def _simple_encode(self, texts: List[str]) -> np.ndarray:
        # Keep vocab fixed between fit/encode to avoid dimension mismatch.
        vocab = self._simple_vocab or {}
        rows = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                idx = vocab.get(tok)
                if idx is None:
                    continue
                rows[i, idx] += 1.0
        return self._normalize(rows)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

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
        elif self.backend == "simple":
            if self._simple_vocab is None:
                logger.warning("Simple vocab missing; skipping vocab save.")
                return
            vocab_list = [token for token, _ in sorted(self._simple_vocab.items(), key=lambda item: item[1])]
            with path.open("w", encoding="utf-8") as f:
                json.dump({"vocab": vocab_list}, f, ensure_ascii=True)

    @classmethod
    def load_from_metadata(cls, meta: EmbeddingMetadata, index_dir: Path) -> "EmbeddingModel":
        if meta.backend == "sentence-transformers" and _try_import_sentence_transformers() is None:
            raise RuntimeError(
                "Index was built with sentence-transformers embeddings, but sentence-transformers is not installed. "
                "Install it or rebuild the index with a different backend."
            )
        model = cls(prefer=meta.backend, model_name=meta.model_name)
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
        if meta.backend == "simple":
            if not meta.vectorizer_path:
                raise RuntimeError("Simple embeddings vocab missing; rebuild the index.")
            vocab_path = Path(meta.vectorizer_path)
            if not vocab_path.is_absolute():
                vocab_path = index_dir / vocab_path
            if not vocab_path.exists():
                raise RuntimeError(f"Simple embeddings vocab file missing at {vocab_path}. Rebuild the index.")
            data = json.loads(vocab_path.read_text(encoding="utf-8"))
            vocab_list = data.get("vocab", []) if isinstance(data, dict) else []
            model._simple_vocab = {token: idx for idx, token in enumerate(vocab_list)}
        return model
