import io

import numpy as np
import pytest
from PIL import Image

from app.control.image_analysis_service import (DetectedObject, ImageAnalysisService,
                                                 validate_image_file)
from app.control.speech_to_text_service import SpeechNotRecognized, SpeechToTextService
from app.control.upload_store import UploadStore
from tests.fakes import FakeDetector, FakeImageEmbedder, FakeRecognizer, unit

MSG = "Only JPG/PNG, max 10MB"


def _img_bytes(fmt, size=(64, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, format=fmt)
    return buf.getvalue()


# --- validate_image_file -------------------------------------------------
def test_png_and_jpeg_accepted():
    assert validate_image_file(_img_bytes("PNG")) == "png"
    assert validate_image_file(_img_bytes("JPEG")) == "jpg"


def test_gif_renamed_as_png_rejected():
    with pytest.raises(ValueError, match=MSG):
        validate_image_file(_img_bytes("GIF"))


def test_text_file_rejected():
    with pytest.raises(ValueError, match=MSG):
        validate_image_file(b"hello, not an image")


def test_over_10mb_rejected():
    data = _img_bytes("PNG") + b"\0" * (10 * 1024 * 1024)
    with pytest.raises(ValueError, match=MSG):
        validate_image_file(data)


# --- UploadStore ---------------------------------------------------------
def test_upload_store_round_trip(tmp_path):
    store = UploadStore(tmp_path)
    uid = store.save(b"abc", "png")
    assert store.path(uid).read_bytes() == b"abc"


@pytest.mark.parametrize("bad", ["../x", "nothex", "0" * 32])
def test_upload_store_rejects_bad_or_missing_ids(tmp_path, bad):
    with pytest.raises(KeyError):
        UploadStore(tmp_path).path(bad)


# --- SpeechToTextService -------------------------------------------------
def test_empty_transcript_not_recognized(tmp_path):
    with pytest.raises(SpeechNotRecognized):
        SpeechToTextService(FakeRecognizer("  ")).convert_speech_to_text(tmp_path / "a.webm")


def test_high_no_speech_prob_not_recognized(tmp_path):
    with pytest.raises(SpeechNotRecognized):
        SpeechToTextService(FakeRecognizer("hmm", 0.7)).convert_speech_to_text(tmp_path / "a.webm")


def test_clear_speech_returns_text_and_language(tmp_path):
    text, lang = SpeechToTextService(FakeRecognizer(" điện thoại ", 0.2)).convert_speech_to_text(
        tmp_path / "a.webm")
    assert (text, lang) == ("điện thoại", "vi")


# --- ImageAnalysisService ------------------------------------------------
def _svc(objects=None, vec=None):
    return ImageAnalysisService(FakeDetector(objects or []), FakeImageEmbedder(vec if vec is not None else unit(0)))


def test_preprocess_limits_long_side():
    img = _svc().preprocess_image(_img_bytes("PNG", (3000, 1000)))
    assert img.size == (1024, 341)
    assert img.mode == "RGB"


def test_detect_objects_filters_low_confidence_and_sorts():
    objs = [DetectedObject("cup", 0.2, (0, 0, 5, 5)), DetectedObject("shoe", 0.5, (0, 0, 9, 9)),
            DetectedObject("bag", 0.9, (1, 1, 8, 8))]
    found = _svc(objs).detect_objects(Image.new("RGB", (10, 10)))
    assert [o.label for o in found] == ["bag", "shoe"]


def test_detect_objects_caps_at_eight():
    objs = [DetectedObject(f"o{i}", 0.9, (0, 0, 5, 5)) for i in range(12)]
    assert len(_svc(objs).detect_objects(Image.new("RGB", (10, 10)))) == 8


def test_extract_features_clamps_box():
    emb = FakeImageEmbedder(unit(0))
    svc = ImageAnalysisService(FakeDetector([]), emb)
    feats = svc.extract_features(Image.new("RGB", (100, 50)), (-50, -50, 99999, 99999))
    assert feats.shape == (512,)
    assert emb.calls[0].size == (100, 50)


def test_extract_features_zero_area_box_uses_full_image():
    emb = FakeImageEmbedder(unit(0))
    svc = ImageAnalysisService(FakeDetector([]), emb)
    svc.extract_features(Image.new("RGB", (100, 50)), (10, 10, 10, 10))
    assert emb.calls[0].size == (100, 50)


def test_extract_features_crops_box():
    emb = FakeImageEmbedder(unit(0))
    svc = ImageAnalysisService(FakeDetector([]), emb)
    svc.extract_features(Image.new("RGB", (100, 50)), (10, 5, 40, 25))
    assert emb.calls[0].size == (30, 20)
    assert np.isclose(np.linalg.norm(svc.extract_features(Image.new("RGB", (10, 10)), None)), 1.0)
