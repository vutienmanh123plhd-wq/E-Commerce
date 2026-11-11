"""Test doubles for the AI models, so tests never download a model."""
import numpy as np

from app.control.image_analysis_service import DetectedObject
from app.control.text_utils import normalize_keyword

DIM = 512


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


class FakeTextEmbedder:
    def __init__(self, mapping: dict[str, np.ndarray] | None = None, default: np.ndarray | None = None):
        self.mapping = {normalize_keyword(k): v for k, v in (mapping or {}).items()}
        self.default = default if default is not None else unit(DIM - 1)

    def encode_text(self, texts):
        return np.stack([self.mapping.get(normalize_keyword(t), self.default) for t in texts])


class FailingTextEmbedder:
    def encode_text(self, texts):
        raise RuntimeError("model crashed")


class FakeImageEmbedder:
    def __init__(self, vec: np.ndarray):
        self.vec = vec
        self.calls = []

    def encode_image(self, images):
        self.calls.extend(images)
        return np.stack([self.vec for _ in images])


class FakeRecognizer:
    def __init__(self, text: str, no_speech_prob: float = 0.1, language: str = "vi"):
        self.result = {"text": text, "no_speech_prob": no_speech_prob, "language": language}

    def transcribe(self, path):
        return dict(self.result)


class FakeDetector:
    def __init__(self, objects: list[DetectedObject]):
        self.objects = objects

    def detect(self, img):
        return list(self.objects)
