import numpy as np
import pytest

from app.control.search_service import SearchService
from app.data_access.product_repository import ProductRepository
from app.data_access.search_history_repository import SearchHistoryRepository
from app.data_access.vector_index import VectorIndex
from app.domain.models import Category, Product, SearchHistory
from tests.fakes import FailingTextEmbedder, FakeImageEmbedder, FakeTextEmbedder, unit


@pytest.fixture
def session(session_factory):
    with session_factory() as s:
        shoes = Category(category_id=1, name="mens-shoes")
        tech = Category(category_id=2, name="laptops")
        s.add_all([
            Product(product_id=1, name="Running Shoes", description="Light", price=50, stock=5,
                    category=shoes),
            Product(product_id=2, name="Gaming Laptop", description="Fast", price=900, stock=3,
                    category=tech),
            Product(product_id=3, name="Smart Phone", description="5G phone", price=500, stock=3,
                    category=Category(category_id=3, name="smartphones")),
        ])
        s.commit()
        yield s


def _index():
    # product 1 ~ unit(1), product 2 ~ unit(2), product 3 ~ unit(3)
    return VectorIndex([1, 2, 3], np.stack([unit(1), unit(2), unit(3)]),
                       np.stack([unit(11), unit(12), unit(13)]))


def _service(session, index=None, text=None, image=None, translator=None):
    # semantic_min is low because a 3-product index cannot produce large z-scores
    return SearchService(ProductRepository(session), SearchHistoryRepository(session),
                         index if index is not None else _index(), text, image,
                         translator=translator, semantic_min=0.5)


class FakeTranslator:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        return self.mapping.get(text, text)


def test_vietnamese_query_matches_translated_keyword(session):
    session.add(Product(product_id=4, name="Red Lipstick", description="Matte", price=12, stock=9))
    session.commit()
    tr = FakeTranslator({"son môi": "lipstick."})
    res = _service(session, text=FakeTextEmbedder(), translator=tr).find_by_keyword("son môi")
    assert [r.name for r in res] == ["Red Lipstick"]


def test_translation_does_not_match_descriptions(session):
    session.add(Product(product_id=4, name="Floral Perfume", description="fruity notes",
                        price=40, stock=2))
    session.commit()
    tr = FakeTranslator({"trái cây": "fruit"})
    res = _service(session, text=FakeTextEmbedder(), translator=tr).find_by_keyword("trái cây")
    assert res == []


def test_repetitive_translation_is_ignored(session):
    tr = FakeTranslator({"dầu ăn": "Hey, hey, hey, hey, hey, hey"})
    session.add(Product(product_id=4, name="Hey Hey Speaker", description="", price=1, stock=1))
    session.commit()
    res = _service(session, text=FakeTextEmbedder(), translator=tr).find_by_keyword("dầu ăn")
    assert res == []


def test_ascii_query_is_not_translated(session):
    tr = FakeTranslator({})
    _service(session, text=FakeTextEmbedder(), translator=tr).find_by_keyword("laptop")
    assert tr.calls == []


def test_failing_translator_still_searches(session):
    class Broken:
        def translate(self, text):
            raise RuntimeError("down")
    text = FakeTextEmbedder({"giày chạy bộ": unit(1)})
    res = _service(session, text=text, translator=Broken()).find_by_keyword("giày chạy bộ")
    assert [r.name for r in res] == ["Running Shoes"]


def test_exact_keyword_match_comes_first(session):
    text = FakeTextEmbedder({"laptop": unit(3)})  # semantic says phone, keyword says laptop
    res = _service(session, text=text).find_by_keyword("laptop")
    assert [r.name for r in res] == ["Gaming Laptop", "Smart Phone"]
    assert res[0].score == 1.0


def test_vietnamese_query_found_semantically(session):
    text = FakeTextEmbedder({"giày chạy bộ": unit(1)})
    res = _service(session, text=text).find_by_keyword("giày chạy bộ")
    assert [r.name for r in res] == ["Running Shoes"]


def test_vietnamese_query_case_and_spaces_same_result(session):
    text = FakeTextEmbedder({"giày chạy bộ": unit(1)})
    res = _service(session, text=text).find_by_keyword("Giày  CHẠY bộ")
    assert [r.name for r in res] == ["Running Shoes"]


def test_empty_index_falls_back_to_keyword_only(session):
    empty = VectorIndex.load("does-not-exist.npz")
    res = _service(session, index=empty, text=FakeTextEmbedder()).find_by_keyword("phone")
    assert [r.name for r in res] == ["Smart Phone"]


def test_failing_embedder_falls_back_to_keyword_only(session):
    res = _service(session, text=FailingTextEmbedder()).find_by_keyword("laptop")
    assert [r.name for r in res] == ["Gaming Laptop"]


def test_nonsense_returns_nothing(session):
    assert _service(session, text=FakeTextEmbedder()).find_by_keyword("qwxz") == []


def test_invalid_keyword_raises(session):
    with pytest.raises(ValueError, match="Keyword must be 2-50 characters"):
        _service(session, text=FakeTextEmbedder()).find_by_keyword("a")


def test_image_features_closest_first_and_threshold(session):
    feats = unit(2) * 0.9 + unit(3) * 0.1
    res = _service(session).find_similar_products(feats)
    assert [r.id for r in res] == [2]


def _img_service(session):
    # product 1 ~ unit(1); product 2 is 0.80 similar to product 1; product 3 unrelated
    p2 = 0.8 * unit(1) + 0.6 * unit(2)
    index = VectorIndex([1, 2, 3], np.stack([unit(1), p2, unit(3)]),
                        np.stack([unit(11), unit(12), unit(13)]))
    return _service(session, index=index)


def _catalogue_service(session, n=30):
    """n products with random image/text vectors, like a real catalogue."""
    for i in range(4, n + 1):
        session.add(Product(product_id=i, name=f"Item {i}", description="", price=1, stock=1))
    session.commit()
    rng = np.random.default_rng(7)
    img = rng.normal(size=(n, 512)).astype(np.float32)
    txt = rng.normal(size=(n, 512)).astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    return _service(session, index=VectorIndex(list(range(1, n + 1)), img, txt)), img, txt


def test_real_photo_found_through_image_and_text_agreement(session):
    # a photo of product 5 taken elsewhere: only ~0.6 cosine to its catalogue image,
    # but close to both its image and its text description
    svc, img, txt = _catalogue_service(session)
    noise = np.random.default_rng(1).normal(size=512).astype(np.float32)
    photo = img[4] + txt[4] + 0.6 * noise / np.linalg.norm(noise)
    res = svc.find_similar_products(photo)
    assert res and res[0].id == 5
    assert res[0].score < 0.84  # shown similarity is the honest image cosine


def test_near_duplicate_catalogue_image_is_found(session):
    svc, img, _ = _catalogue_service(session)
    res = svc.find_similar_products(img[9])
    assert res[0].id == 10
    assert res[0].score > 0.99


def test_unrelated_photo_returns_nothing(session):
    svc, _, _ = _catalogue_service(session)
    junk = np.random.default_rng(42).normal(size=512).astype(np.float32)
    assert svc.find_similar_products(junk) == []


def test_image_without_strong_match_returns_nothing(session):
    weak = 0.82 * unit(1) + np.sqrt(1 - 0.82 ** 2) * unit(5)  # best cosine 0.82 < gate 0.84
    assert _img_service(session).find_similar_products(weak) == []


def test_save_search_history_records_type(session):
    svc = _service(session)
    svc.save_search_history(1, "KEYWORD", "laptop")
    row = session.query(SearchHistory).one()
    assert (row.search_type, row.query_text) == ("KEYWORD", "laptop")
