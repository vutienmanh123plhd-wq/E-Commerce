"""SpeechToTextService (control): converts recorded audio to text with Whisper."""
import re
import subprocess
import threading
import unicodedata
from pathlib import Path
from typing import Protocol

import numpy as np

from app import config

SAMPLE_RATE = 16000  # what Whisper expects
LANGUAGES = ("vi", "en")  # the only languages voice search accepts


def pick_language(probs: dict) -> str:
    """Most likely of Vietnamese and English; Whisper's other 97 languages are ignored."""
    return max(LANGUAGES, key=lambda lang: probs.get(lang, 0.0)) if probs else LANGUAGES[0]


def clean_transcript(text: str) -> str:
    """Drop letters of non-Latin scripts (Whisper sometimes invents CJK words in silences)."""
    kept = "".join(c for c in text
                   if not unicodedata.category(c).startswith("L")
                   or "LATIN" in unicodedata.name(c, ""))
    return re.sub(r"\s+", " ", kept).strip()


class SpeechNotRecognized(Exception):
    pass


def decode_audio(path: Path) -> np.ndarray:
    """Decode any browser recording (webm/ogg/wav/...) to 16 kHz mono float32 samples.

    Uses the ffmpeg binary bundled with imageio-ffmpeg, so no system-wide ffmpeg
    is needed (Whisper's own loader calls `ffmpeg` from PATH, absent on Windows).
    Raises ValueError when the file cannot be decoded.
    """
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", str(path),
           "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(f"cannot decode audio: {proc.stderr.decode(errors='replace')[-200:]}")
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


class SpeechRecognizer(Protocol):
    def transcribe(self, path: Path) -> dict: ...


class WhisperRecognizer:
    """OpenAI Whisper. Its log-Mel front end and voice-activity probability play the role of
    noiseReduction(), and its built-in language identification is detectLanguage().

    Decoding installs hooks on the shared model, so concurrent calls are serialized."""

    _lock = threading.Lock()

    def __init__(self, model: str = config.WHISPER_MODEL, device: str | None = None):
        import whisper
        from app.control.embedding import pick_device
        self.device = device or pick_device()
        self.model = whisper.load_model(model, device=self.device,
                                        download_root=str(config.MODELS_DIR / "whisper"))

    def _language_probs(self, audio: np.ndarray) -> dict:
        """Whisper's language probabilities for the first 30 s of the recording."""
        import whisper
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), self.model.dims.n_mels)
        _, probs = self.model.detect_language(mel.to(self.model.device))
        return probs

    def transcribe(self, path: Path) -> dict:
        try:
            audio = decode_audio(path)
        except ValueError as e:
            raise SpeechNotRecognized(config.MSG_SPEAK_AGAIN) from e
        with self._lock:
            language = pick_language(self._language_probs(audio))  # detectLanguage(): vi or en
            result = self.model.transcribe(audio, language=language, fp16=self.device == "cuda",
                                           task="transcribe", condition_on_previous_text=False)
        result["text"] = clean_transcript(result.get("text", ""))
        result["language"] = language
        segments = result.get("segments") or []
        # speech is present if ANY segment is confidently speech (a quiet start must not
        # reject the whole recording); no segments at all means nothing was heard
        no_speech = min((s.get("no_speech_prob", 0.0) for s in segments), default=1.0)
        return {"text": result.get("text", ""), "language": result.get("language", ""),
                "no_speech_prob": no_speech}


class SpeechToTextService:
    def __init__(self, recognizer: SpeechRecognizer):
        self.recognizer = recognizer

    def convert_speech_to_text(self, audio_path: Path) -> tuple[str, str]:
        """convertSpeechToText(audio) -> (recognizedText, language); raises SpeechNotRecognized."""
        result = self.recognizer.transcribe(audio_path)
        text = (result.get("text") or "").strip()
        if not text or result.get("no_speech_prob", 0.0) > config.NO_SPEECH_MAX:
            raise SpeechNotRecognized(config.MSG_SPEAK_AGAIN)
        return text, result.get("language", "")
