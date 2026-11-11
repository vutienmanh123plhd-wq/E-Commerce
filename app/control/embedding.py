"""Text and image encoders (MultimodalEncoder / EmbeddingService in the design).

Both encoders map into the same 512-dimensional CLIP space, so a Vietnamese or
English text query can be compared with product images and product texts.
Heavy libraries are imported lazily so tests run without them.
"""
from typing import Protocol

import numpy as np

TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMAGE_MODEL = "clip-ViT-B-32"


class TextEmbedder(Protocol):
    def encode_text(self, texts: list[str]) -> np.ndarray: ...


class ImageEmbedder(Protocol):
    def encode_image(self, images: list) -> np.ndarray: ...


def pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class ClipTextEmbedder:
    """Multilingual text encoder aligned with CLIP ViT-B/32 (supports Vietnamese)."""

    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(TEXT_MODEL, device=device or pick_device())

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                 batch_size=32).astype(np.float32)


class ClipImageEmbedder:
    """CLIP ViT-B/32 image encoder."""

    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(IMAGE_MODEL, device=device or pick_device())

    def encode_image(self, images: list) -> np.ndarray:
        return self.model.encode(images, normalize_embeddings=True, convert_to_numpy=True,
                                 batch_size=32).astype(np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """English CLIP text encoder (used for product texts when seeding)."""
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                 batch_size=32).astype(np.float32)
