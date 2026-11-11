import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.control.image_analysis_service import DetectedObject
from app.control.search_controller import AppServices
from app.control.upload_store import UploadStore
from app.data_access.vector_index import VectorIndex
from app.domain.models import (Category, ImageSearchQuery, Product, SearchHistory,
                               VoiceSearchQuery)
from app.main import create_app
from tests.fakes import FakeDetector, FakeImageEmbedder, FakeRecognizer, FakeTextEmbedder, unit


@pytest.fixture
def make_client(session_factory, tmp_path):
    with session_factory() as s:
        tech = Category(category_id=1, name="laptops")
        shoes = Category(category_id=2, name="mens-shoes")
        s.add_all([
            Product(product_id=1, name="Gaming Laptop", description="Fast", price=900, stock=3,
                    image_url="/images/1.jpg", category=tech),
            Product(product_id=2, name="Running Shoes", description="Light", price=50, stock=5,
                    image_url="/images/2.jpg", category=shoes),
        ])
        s.commit()
    index = VectorIndex([1, 2], np.stack([unit(1), unit(2)]), np.stack([unit(11), unit(12)]))

    def _make(**over):
        services = AppServices(
            session_factory=session_factory, index=index,
            text_embedder=FakeTextEmbedder({"điện thoại": unit(1)}),
            image_embedder=FakeImageEmbedder(unit(2)),
            recognizer=FakeRecognizer("điện thoại", 0.1),
            detector=FakeDetector([DetectedObject("shoe", 0.9, (2, 2, 20, 20))]),
            uploads=UploadStore(tmp_path / "uploads"),
            semantic_min=0.4,  # a 2-product index cannot produce large z-scores
        )
        for k, v in over.items():
            setattr(services, k, v)
        return TestClient(create_app(services)), session_factory
    return _make


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (0, 0, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _gif():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="GIF")
    return buf.getvalue()


def _count(factory, model):
    with factory() as s:
        return s.query(model).count()


def test_frontend_files_are_revalidated(make_client):
    client, _ = make_client()
    r = client.get("/search_ui.js")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"


# --- keyword --------------------------------------------------------------
def test_suggestions_empty_then_populated(make_client):
    client, _ = make_client()
    assert client.get("/api/suggestions").json() == {"recent": [], "trending": []}
    client.get("/api/search/keyword", params={"q": "laptop"})
    body = client.get("/api/suggestions").json()
    assert body == {"recent": ["laptop"], "trending": ["laptop"]}


def test_keyword_too_short_400(make_client):
    client, _ = make_client()
    r = client.get("/api/search/keyword", params={"q": "a"})
    assert r.status_code == 400
    assert r.json() == {"detail": "Keyword must be 2-50 characters"}


def test_keyword_found_saves_history(make_client):
    client, factory = make_client()
    r = client.get("/api/search/keyword", params={"q": "laptop"})
    assert r.status_code == 200
    assert [p["name"] for p in r.json()["products"]] == ["Gaming Laptop"]
    assert _count(factory, SearchHistory) == 1


def test_keyword_nothing_found_no_history(make_client):
    client, factory = make_client()
    r = client.get("/api/search/keyword", params={"q": "qwxz"})
    assert r.json() == {"products": []}
    assert _count(factory, SearchHistory) == 0


def test_product_detail_and_404(make_client):
    client, _ = make_client()
    assert client.get("/api/products/2").json()["name"] == "Running Shoes"
    assert client.get("/api/products/999").status_code == 404


# --- voice ----------------------------------------------------------------
def test_transcribe_returns_text(make_client):
    client, _ = make_client()
    r = client.post("/api/voice/transcribe", files={"audio": ("a.webm", b"fake", "audio/webm")})
    body = r.json()
    assert r.status_code == 200
    assert body["text"] == "điện thoại" and body["language"] == "vi" and len(body["audio_id"]) == 32


def test_transcribe_unrecognized_422(make_client):
    client, _ = make_client(recognizer=FakeRecognizer("", 0.9))
    r = client.post("/api/voice/transcribe", files={"audio": ("a.webm", b"fake", "audio/webm")})
    assert r.status_code == 422
    assert r.json() == {"detail": "Please speak again"}


def test_voice_search_unknown_audio_404(make_client):
    client, _ = make_client()
    r = client.post("/api/search/voice", json={"text": "laptop", "audio_id": "0" * 32})
    assert r.status_code == 404
    assert r.json() == {"detail": "Upload expired, please try again"}


def test_voice_search_saves_voice_history(make_client):
    client, factory = make_client()
    audio_id = client.post("/api/voice/transcribe",
                           files={"audio": ("a.webm", b"fake", "audio/webm")}).json()["audio_id"]
    r = client.post("/api/search/voice", json={"text": "điện thoại", "audio_id": audio_id})
    assert [p["name"] for p in r.json()["products"]] == ["Gaming Laptop"]
    with factory() as s:
        h = s.query(SearchHistory).one()
        assert (h.search_type, h.query_text) == ("VOICE", "điện thoại")
        assert s.query(VoiceSearchQuery).one().recognized_text == "điện thoại"


def test_voice_unavailable_503(make_client):
    client, _ = make_client(recognizer=None)
    r = client.post("/api/voice/transcribe", files={"audio": ("a.webm", b"fake", "audio/webm")})
    assert r.status_code == 503
    assert r.json() == {"detail": "Voice/Image search is unavailable, please type a keyword"}


# --- image ----------------------------------------------------------------
def test_analyze_gif_400(make_client):
    client, _ = make_client()
    r = client.post("/api/image/analyze", files={"image": ("x.png", _gif(), "image/png")})
    assert r.status_code == 400
    assert r.json() == {"detail": "Only JPG/PNG, max 10MB"}


def test_analyze_png_returns_objects(make_client):
    client, _ = make_client()
    body = client.post("/api/image/analyze", files={"image": ("x.png", _png(), "image/png")}).json()
    assert (body["width"], body["height"]) == (40, 30)
    assert body["objects"] == [{"label": "shoe", "confidence": 0.9, "box": [2, 2, 20, 20]}]
    assert len(body["image_id"]) == 32


def test_image_search_saves_image_history(make_client):
    client, factory = make_client()
    image_id = client.post("/api/image/analyze",
                           files={"image": ("x.png", _png(), "image/png")}).json()["image_id"]
    r = client.post("/api/search/image", json={"image_id": image_id, "box": [2, 2, 20, 20]})
    assert [p["name"] for p in r.json()["products"]] == ["Running Shoes"]
    with factory() as s:
        assert s.query(SearchHistory).one().search_type == "IMAGE"
        assert s.query(ImageSearchQuery).one().image_path.endswith(".png")


def test_image_search_no_match_empty(make_client):
    client, factory = make_client(image_embedder=FakeImageEmbedder(unit(400)))
    image_id = client.post("/api/image/analyze",
                           files={"image": ("x.png", _png(), "image/png")}).json()["image_id"]
    r = client.post("/api/search/image", json={"image_id": image_id})
    assert r.json() == {"products": []}
    assert _count(factory, SearchHistory) == 0


def test_image_unavailable_503(make_client):
    client, _ = make_client(image_embedder=None)
    r = client.post("/api/image/analyze", files={"image": ("x.png", _png(), "image/png")})
    assert r.status_code == 503
