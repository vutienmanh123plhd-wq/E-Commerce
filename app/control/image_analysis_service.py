"""ImageAnalysisService (control): validates, pre-processes and analyses product images."""
import io
import threading
import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

from app import config


@dataclass
class DetectedObject:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels of the preprocessed image


class ObjectDetector(Protocol):
    def detect(self, img: Image.Image) -> list[DetectedObject]: ...


def validate_image_file(data: bytes) -> str:
    """validateFileFormat(image): JPEG/PNG by magic bytes and at most 10 MB."""
    if len(data) > config.MAX_IMAGE_BYTES:
        raise ValueError(config.MSG_INVALID_IMAGE)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    raise ValueError(config.MSG_INVALID_IMAGE)


class YoloDetector:
    """YOLOv8n object detector (ultralytics). The model is shared by all requests,
    and its predictor is not thread-safe, so calls are serialized."""

    _lock = threading.Lock()

    def __init__(self, device: str | None = None):
        from ultralytics import YOLO
        from app.control.embedding import pick_device
        self.device = device or pick_device()
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(config.MODELS_DIR / "yolov8n.pt"))

    def detect(self, img: Image.Image) -> list[DetectedObject]:
        with self._lock:
            result = self.model.predict(img, device=self.device, verbose=False,
                                        conf=config.YOLO_CONF)[0]
        names = result.names
        return [DetectedObject(names[int(c)], float(p), tuple(int(v) for v in b))
                for b, p, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                   result.boxes.cls.tolist())]


class ImageAnalysisService:
    def __init__(self, detector: ObjectDetector | None, image_embedder):
        self.detector = detector
        self.image_embedder = image_embedder

    def preprocess_image(self, data: bytes) -> Image.Image:
        """preprocessImage(): fix EXIF rotation, convert to RGB, long side <= 1024.

        Raises ValueError for a body that cannot be decoded or declares too many pixels.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                img = Image.open(io.BytesIO(data))
                if img.width * img.height > config.MAX_IMAGE_PIXELS:
                    raise ValueError(config.MSG_INVALID_IMAGE)
                img = ImageOps.exif_transpose(img).convert("RGB")
        except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning,
                SyntaxError) as e:
            raise ValueError(config.MSG_INVALID_IMAGE) from e
        img.thumbnail((config.MAX_SIDE, config.MAX_SIDE))
        return img

    def detect_objects(self, img: Image.Image) -> list[DetectedObject]:
        """detectObjects(): confident objects, best first, at most 8."""
        if self.detector is None:
            return []
        found = [o for o in self.detector.detect(img) if o.confidence >= config.YOLO_CONF]
        return sorted(found, key=lambda o: -o.confidence)[:config.YOLO_MAX]

    def extract_features(self, img: Image.Image, box: tuple | list | None) -> np.ndarray:
        """extractFeatures(image): CLIP vector of the selected object (or the whole image)."""
        target = img
        if box is not None:
            w, h = img.size
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
            y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                target = img.crop((x1, y1, x2, y2))
        vec = self.image_embedder.encode_image([target])[0]
        return vec / (np.linalg.norm(vec) or 1.0)
