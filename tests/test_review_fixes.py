"""Regression tests for the final code review findings."""
import io
import struct
import threading
import time
import zlib

import pytest
from PIL import Image

from app.control.image_analysis_service import YoloDetector
from app.control.speech_to_text_service import WhisperRecognizer
from app.data_access.product_repository import ProductRepository
from app.data_access.search_history_repository import SearchHistoryRepository
from app.domain.models import SearchHistory
from tests.test_api import make_client  # noqa: F401  (fixture)
from tests.test_audio_decoding import _wav


def _png(size=(40, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, (0, 0, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _png_header_claiming(w, h):
    """A tiny PNG whose header declares w x h pixels (a decompression bomb)."""
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0" * 64)) \
        + chunk(b"IEND", b"")


# Finding 1: corrupt image body / decompression bomb -> 400, not 500
def test_truncated_png_is_rejected_with_400(make_client):
    client, _ = make_client()
    r = client.post("/api/image/analyze", files={"image": ("x.png", _png()[:40], "image/png")})
    assert r.status_code == 400
    assert r.json() == {"detail": "Only JPG/PNG, max 10MB"}


def test_decompression_bomb_is_rejected_with_400(make_client):
    client, _ = make_client()
    r = client.post("/api/image/analyze",
                    files={"image": ("x.png", _png_header_claiming(20000, 20000), "image/png")})
    assert r.status_code == 400


# Finding 2: oversized uploads are refused
def test_oversized_audio_is_rejected(make_client):
    client, _ = make_client()
    r = client.post("/api/voice/transcribe",
                    files={"audio": ("a.webm", b"\0" * (10 * 1024 * 1024 + 1), "audio/webm")})
    assert r.status_code == 400
    assert r.json() == {"detail": "Recording too long, please try again"}


# Finding 3: client-chosen extensions are never stored
def test_audio_upload_never_stored_as_html(make_client, tmp_path):
    client, _ = make_client()
    client.post("/api/voice/transcribe", files={"audio": ("x.html", b"<script>", "text/html")})
    assert not list((tmp_path / "uploads").glob("*.html"))
    assert len(list((tmp_path / "uploads").glob("*.webm"))) == 1


# Finding 4: shared models are used by one request at a time
class _SlowModel:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def _work(self, result):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return result

    def transcribe(self, *a, **k):
        return self._work({"text": "x", "language": "vi", "segments": []})


def _run_parallel(fn, n=4):
    threads = [threading.Thread(target=fn) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_whisper_calls_are_serialized(tmp_path):
    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec.model, rec.device = _SlowModel(), "cpu"
    rec._language_probs = lambda audio: {"vi": 0.9}
    wav = _wav(tmp_path / "a.wav", seconds=0.2)
    _run_parallel(lambda: rec.transcribe(wav))
    assert rec.model.peak == 1


def test_yolo_calls_are_serialized():
    class Boxes:
        xyxy = conf = cls = type("T", (), {"tolist": lambda self: []})()

    slow = _SlowModel()
    det = YoloDetector.__new__(YoloDetector)
    det.device = "cpu"
    det.model = type("M", (), {"predict": lambda self, *a, **k: slow._work(
        [type("R", (), {"names": {}, "boxes": Boxes()})()])})()
    _run_parallel(lambda: det.detect(Image.new("RGB", (8, 8))))
    assert slow.peak == 1


# Finding 5: suggestions are de-duplicated case- and space-insensitively
def test_suggestions_deduplicate_case_and_spaces(session_factory):
    with session_factory() as s:
        repo = SearchHistoryRepository(s)
        for q in ["Laptop", "laptop", "giày  chạy", "Giày chạy", "phone"]:
            repo.save(SearchHistory(customer_id=1, search_type="KEYWORD", query_text=q))
        assert repo.get_recent_searches(1, 5) == ["phone", "Giày chạy", "laptop"]
        assert ProductRepository(s).get_trending_keywords(5) == ["Giày chạy", "laptop", "phone"]


def test_keyword_history_saved_with_collapsed_spaces(make_client):
    client, factory = make_client()
    client.get("/api/search/keyword", params={"q": "  gaming   laptop "})
    with factory() as s:
        assert s.query(SearchHistory).one().query_text == "gaming laptop"
