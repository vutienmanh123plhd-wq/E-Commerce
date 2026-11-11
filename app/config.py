"""Paths, limits and thresholds shared by all layers."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / ".models"
# Keep downloaded model weights inside the project (drive C: may be full).
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS_DIR / "ultralytics"))
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "shop.db"
DB_URL = f"sqlite:///{DB_PATH}"
IMAGES_DIR = DATA_DIR / "images"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTORS_PATH = DATA_DIR / "vectors.npz"
STATIC_DIR = ROOT_DIR / "app" / "presentation" / "static"

DEMO_CUSTOMER_ID = 1

# Validation (BR01, BR05)
KEYWORD_MIN = 2
KEYWORD_MAX = 50
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000          # refuse decompression bombs before decoding
MAX_AUDIO_BYTES = 10 * 1024 * 1024
AUDIO_SUFFIXES = {"webm", "ogg", "wav", "mp3", "m4a"}

# Result limits
KEYWORD_LIMIT = 24
IMAGE_LIMIT = 12

# Similarity thresholds, tuned on the seeded catalogue in the smoke test.
# Text queries: fused z-score 0.5*z(image cos) + 0.5*z(text cos) over all products.
SEMANTIC_MIN_Z = 1.65
# Image queries. Measured on 29 web photos of catalogue product types and 6 junk
# images (blank, noise, gradient, shapes, text):
#  - image-to-image cosine: web photos 0.52-0.84, junk 0.74-0.82 -> cannot separate;
#    only near-duplicates of catalogue pictures reach >= 0.84.
#  - fused z = 0.5*z(image cos) + 0.5*z(text cos): correct web photos mostly 2.1-3.8,
#    junk at most 2.05.
IMAGE_GATE = 0.84     # a near-duplicate of a catalogue picture is always accepted
IMAGE_GATE_Z = 2.1    # otherwise the best product's fused z must reach this
IMAGE_MIN_Z = 1.6     # related products listed after the gate

# AI models
NO_SPEECH_MAX = 0.6
WHISPER_MODEL = "small"
YOLO_CONF = 0.35
YOLO_MAX = 8
MAX_SIDE = 1024
EMBED_DIM = 512

# User-facing messages (copied from the sequence diagrams)
MSG_INVALID_KEYWORD = "Keyword must be 2-50 characters"
MSG_INVALID_IMAGE = "Only JPG/PNG, max 10MB"
MSG_SPEAK_AGAIN = "Please speak again"
MSG_AUDIO_TOO_LONG = "Recording too long, please try again"
MSG_MODEL_UNAVAILABLE = "Voice/Image search is unavailable, please type a keyword"
MSG_UPLOAD_EXPIRED = "Upload expired, please try again"
