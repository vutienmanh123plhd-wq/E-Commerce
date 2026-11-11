"""Keyword validation (BR01) and normalisation (BR02)."""
import re
import unicodedata

from app import config


def normalize_keyword(text: str) -> str:
    """normalizeKeyword(keyword): lower case, no Vietnamese diacritics, single spaces."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip().lower()


def validate_keyword(text: str) -> str:
    """validateFormat(keyword): 2-50 characters after trimming, else ValueError."""
    keyword = (text or "").strip()
    if not config.KEYWORD_MIN <= len(keyword) <= config.KEYWORD_MAX:
        raise ValueError(config.MSG_INVALID_KEYWORD)
    return keyword
