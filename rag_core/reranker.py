from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def _try_import_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder
    except Exception:
        return None


def _device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Reranker:
    _cache: dict[str, "Reranker"] = {}

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.device = _device()
        CrossEncoder = _try_import_cross_encoder()
        if not CrossEncoder:
            logger.warning("sentence-transformers CrossEncoder not available; rerank disabled.")
            return
        try:
            self.model = CrossEncoder(model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load reranker model %s: %s", model_name, exc)
            self.model = None

    @classmethod
    def get(cls, model_name: str) -> "Reranker":
        if model_name not in cls._cache:
            cls._cache[model_name] = cls(model_name)
        return cls._cache[model_name]

    def available(self) -> bool:
        return self.model is not None

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not self.model:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]
