"""Voice search accepts Vietnamese and English only."""
from app.control.speech_to_text_service import (WhisperRecognizer, clean_transcript,
                                                pick_language)
from tests.test_audio_decoding import _wav


def test_pick_language_ignores_other_languages():
    assert pick_language({"zh": 0.6, "vi": 0.3, "en": 0.1}) == "vi"


def test_pick_language_prefers_english_when_more_likely():
    assert pick_language({"ko": 0.5, "en": 0.3, "vi": 0.2}) == "en"


def test_pick_language_defaults_to_vietnamese():
    assert pick_language({}) == "vi"


def test_clean_transcript_drops_non_latin_scripts():
    raw = " Điện thoại Khu n unboxing편 anh hiệu, mình nói 电话 ra"
    assert clean_transcript(raw) == "Điện thoại Khu n unboxing anh hiệu, mình nói ra"


def test_clean_transcript_keeps_english_and_vietnamese():
    assert clean_transcript("iPhone 13 Pro giá rẻ") == "iPhone 13 Pro giá rẻ"


def test_transcribe_forces_the_chosen_language(tmp_path):
    seen = {}

    class Model:
        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)
            return {"text": " 电话 phone", "language": kwargs.get("language"),
                    "segments": [{"no_speech_prob": 0.1}]}

    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec.model, rec.device = Model(), "cpu"
    rec._language_probs = lambda audio: {"zh": 0.7, "en": 0.2, "vi": 0.1}
    result = rec.transcribe(_wav(tmp_path / "a.wav"))
    assert seen["language"] == "en"
    assert result["language"] == "en"
    assert result["text"] == "phone"
