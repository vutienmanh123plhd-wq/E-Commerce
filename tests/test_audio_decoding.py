"""Whisper must not depend on a system-wide ffmpeg (it is not installed on Windows by default)."""
import math
import struct
import wave

import numpy as np
import pytest

from app.control.speech_to_text_service import (SpeechNotRecognized, SpeechToTextService,
                                                WhisperRecognizer, decode_audio)


def _wav(path, seconds=1.0, rate=8000, channels=2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) * channels
            for i in range(int(seconds * rate)))
        w.writeframes(frames)
    return path


def test_decode_audio_returns_16khz_mono_float32(tmp_path):
    audio = decode_audio(_wav(tmp_path / "a.wav"))
    assert audio.dtype == np.float32
    assert abs(len(audio) - 16000) < 400
    assert 0.1 < np.abs(audio).max() <= 1.0


def test_decode_audio_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.webm"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(ValueError):
        decode_audio(bad)


def test_whisper_receives_decoded_samples_not_a_path(tmp_path):
    seen = {}

    class Model:
        def transcribe(self, audio, **kwargs):
            seen["audio"] = audio
            return {"text": "điện thoại", "language": "vi", "segments": [{"no_speech_prob": 0.1}]}

    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec.model, rec.device = Model(), "cpu"
    rec._language_probs = lambda audio: {"vi": 0.9}
    result = rec.transcribe(_wav(tmp_path / "a.wav"))
    assert isinstance(seen["audio"], np.ndarray)
    assert result["text"] == "điện thoại"


def _recognizer_returning(segments):
    class Model:
        def transcribe(self, audio, **kwargs):
            return {"text": " Điện thoại", "language": "vi", "segments": segments}

    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec.model, rec.device = Model(), "cpu"
    rec._language_probs = lambda audio: {"vi": 0.9}
    return rec


def test_quiet_start_does_not_reject_clear_speech_later(tmp_path):
    # measured on a real 53 s recording: early segments 0.74, later ones 0.28
    rec = _recognizer_returning([{"no_speech_prob": 0.74}, {"no_speech_prob": 0.28}])
    text, lang = SpeechToTextService(rec).convert_speech_to_text(_wav(tmp_path / "a.wav"))
    assert (text, lang) == ("Điện thoại", "vi")


def test_all_segments_silent_is_rejected(tmp_path):
    rec = _recognizer_returning([{"no_speech_prob": 0.9}, {"no_speech_prob": 0.7}])
    with pytest.raises(SpeechNotRecognized):
        SpeechToTextService(rec).convert_speech_to_text(_wav(tmp_path / "a.wav"))


def test_undecodable_recording_asks_to_speak_again(tmp_path):
    bad = tmp_path / "bad.webm"
    bad.write_bytes(b"not audio at all")
    rec = WhisperRecognizer.__new__(WhisperRecognizer)
    rec.model, rec.device = object(), "cpu"
    with pytest.raises(SpeechNotRecognized):
        SpeechToTextService(rec).convert_speech_to_text(bad)
