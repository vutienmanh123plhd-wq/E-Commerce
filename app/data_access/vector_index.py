"""VectorIndex: product image/text embeddings with cosine nearest-neighbour search.

Stands in for PostgreSQL + pgvector (Figure 4.7). Only this class would change
to move the index into pgvector.
"""
from pathlib import Path
from typing import Literal

import numpy as np

Space = Literal["image", "text", "both"]


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class VectorIndex:
    def __init__(self, product_ids: list[int], image_vecs: np.ndarray, text_vecs: np.ndarray):
        self.product_ids = [int(i) for i in product_ids]
        self.image_vecs = _normalize_rows(image_vecs) if len(self.product_ids) else np.zeros((0, 0))
        self.text_vecs = _normalize_rows(text_vecs) if len(self.product_ids) else np.zeros((0, 0))

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        path = Path(path)
        if not path.exists():
            return cls([], np.zeros((0, 0)), np.zeros((0, 0)))
        data = np.load(path)
        return cls(data["product_ids"].tolist(), data["image_vecs"], data["text_vecs"])

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, product_ids=np.array(self.product_ids), image_vecs=self.image_vecs,
                 text_vecs=self.text_vecs)

    @property
    def is_empty(self) -> bool:
        return not self.product_ids

    def nearest(self, query: np.ndarray, k: int, min_score: float,
                space: Space) -> list[tuple[int, float]]:
        """Return up to k (product_id, cosine score) pairs with score >= min_score."""
        if self.is_empty:
            return []
        q = _normalize_rows(np.asarray(query).reshape(-1))
        if space == "image":
            scores = self.image_vecs @ q
        elif space == "text":
            scores = self.text_vecs @ q
        else:
            scores = np.maximum(self.image_vecs @ q, self.text_vecs @ q)
        order = np.argsort(-scores)[:k]
        return [(self.product_ids[i], float(scores[i])) for i in order if scores[i] >= min_score]

    def fused_nearest(self, queries: np.ndarray, k: int, min_z: float) -> list[tuple[int, float]]:
        """Text queries: relevance = 0.5*z(image cosine) + 0.5*z(text cosine).

        z-scores are taken over the whole catalogue, so a product must stand out
        from the rest rather than pass an absolute cosine value; with several
        query variants (original + translation) each product keeps its best score.
        """
        if self.is_empty:
            return []
        q = _normalize_rows(np.atleast_2d(queries))
        fused = 0.5 * _zscore(self.image_vecs @ q.T) + 0.5 * _zscore(self.text_vecs @ q.T)
        best = fused.max(axis=1)
        order = np.argsort(-best)[:k]
        return [(self.product_ids[i], float(best[i])) for i in order if best[i] >= min_z]


def _zscore(scores: np.ndarray) -> np.ndarray:
    """Column-wise z-score; a column with no spread scores 0 everywhere."""
    std = scores.std(axis=0, keepdims=True)
    return np.where(std > 1e-6, (scores - scores.mean(axis=0, keepdims=True)) / np.where(std > 1e-6, std, 1), 0.0)
