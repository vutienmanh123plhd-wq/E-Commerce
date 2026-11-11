import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from app.domain.models import SearchHistory, VoiceSearchQuery

TABLES = {
    "customers", "categories", "products", "carts", "cart_items",
    "orders", "order_items", "search_histories", "voice_search_queries", "image_search_queries",
}


def test_all_ten_tables_created(session_factory):
    with session_factory() as s:
        names = set(inspect(s.get_bind()).get_table_names())
    assert names == TABLES


def test_detail_search_id_unique(session_factory):
    with session_factory() as s:
        h = SearchHistory(customer_id=1, search_type="VOICE", query_text="phone")
        s.add(h)
        s.commit()
        s.add(VoiceSearchQuery(search_id=h.search_id, audio_path="a.webm", recognized_text="phone"))
        s.add(VoiceSearchQuery(search_id=h.search_id, audio_path="b.webm", recognized_text="phone"))
        with pytest.raises(IntegrityError):
            s.commit()
