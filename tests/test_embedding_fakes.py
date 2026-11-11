import numpy as np

from app.control.embedding import ImageEmbedder, TextEmbedder, pick_device
from tests.fakes import FakeImageEmbedder, FakeTextEmbedder, unit


def test_fake_text_embedder_matches_protocol_and_shape():
    emb: TextEmbedder = FakeTextEmbedder({"giày": unit(1)})
    out = emb.encode_text(["Giày", "other"])
    assert out.shape == (2, 512)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert out[0][1] == 1.0


def test_fake_image_embedder_shape():
    emb: ImageEmbedder = FakeImageEmbedder(unit(2))
    assert emb.encode_image([object(), object()]).shape == (2, 512)


def test_pick_device_returns_known_value():
    assert pick_device() in {"cuda", "cpu"}
