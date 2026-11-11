import pytest

from app.control.text_utils import normalize_keyword, validate_keyword

MSG = "Keyword must be 2-50 characters"


def test_normalize_removes_vietnamese_diacritics_case_and_spaces():
    assert normalize_keyword("Giày  CHẠY bộ") == "giay chay bo"


def test_normalize_handles_d_with_stroke():
    assert normalize_keyword("Đồng hồ") == "dong ho"


def test_validate_strips_and_accepts_two_chars():
    assert validate_keyword(" ab ") == "ab"


def test_validate_accepts_fifty_chars():
    assert validate_keyword("x" * 50) == "x" * 50


@pytest.mark.parametrize("bad", ["a", "   ", "x" * 51, ""])
def test_validate_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match=MSG):
        validate_keyword(bad)
