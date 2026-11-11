"""Optional Vietnamese -> English query translation (helps the CLIP encoders and exact matching)."""
import re
from typing import Protocol

from app.control.text_utils import normalize_keyword

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"


class Translator(Protocol):
    def translate(self, text: str) -> str: ...


def has_vietnamese(text: str) -> bool:
    """True when the text contains Vietnamese letters with diacritics."""
    return normalize_keyword(text) != re.sub(r"\s+", " ", text).strip().lower()


def clean_translation(original: str, translated: str) -> str | None:
    """Return a usable English query, or None for an empty, unchanged or degenerate result."""
    text = re.sub(r"[^\w\s'-]", " ", translated or "").strip()
    text = re.sub(r"\s+", " ", text)
    words = text.lower().split()
    if not words or normalize_keyword(text) == normalize_keyword(original):
        return None
    if len(words) > 2 * len(original.split()) + 2:  # rambling output
        return None
    if max(words.count(w) for w in words) > 2:  # "hey, hey, hey, ..."
        return None
    return text


class MarianTranslator:
    def __init__(self, device: str | None = None):
        from transformers import MarianMTModel, MarianTokenizer
        from app.control.embedding import pick_device
        self.device = device or pick_device()
        self.tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
        self.model = MarianMTModel.from_pretrained(TRANSLATION_MODEL).to(self.device)

    def translate(self, text: str) -> str:
        batch = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**batch, max_new_tokens=24, num_beams=4)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
