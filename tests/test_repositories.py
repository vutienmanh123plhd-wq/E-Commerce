import pytest

from app.data_access.product_repository import ProductRepository
from app.data_access.search_history_repository import SearchHistoryRepository
from app.domain.models import Category, ImageSearchQuery, Product, SearchHistory


@pytest.fixture
def session(session_factory):
    with session_factory() as s:
        shoes = Category(category_id=1, name="mens-shoes")
        tech = Category(category_id=2, name="laptops")
        home = Category(category_id=3, name="home-decoration")
        s.add_all([shoes, tech, home])
        s.add_all([
            Product(product_id=1, name="Running Shoes", description="Light runner", price=50,
                    stock=5, category=shoes),
            Product(product_id=2, name="Gaming Laptop", description="Fast", price=900,
                    stock=3, category=tech),
            Product(product_id=3, name="Wooden Rack", description="Rack that holds shoes",
                    price=30, stock=2, category=home),
            Product(product_id=4, name="Leather Boot", description="Warm", price=80,
                    stock=1, category=shoes),
        ])
        s.commit()
        yield s


def _names(products):
    return [p.name for p in products]


def test_find_by_keyword_name_before_category_before_description(session):
    found = ProductRepository(session).find_by_keyword("shoe", limit=24)
    assert _names(found) == ["Running Shoes", "Leather Boot", "Wooden Rack"]


def test_find_by_keyword_matches_unaccented_name(session):
    session.add(Product(product_id=5, name="Áo sơ mi", description="", price=10, stock=1))
    session.commit()
    assert _names(ProductRepository(session).find_by_keyword("ao so mi", limit=24)) == ["Áo sơ mi"]


def test_find_by_keyword_can_skip_descriptions(session):
    found = ProductRepository(session).find_by_keyword("shoe", limit=24, include_description=False)
    assert _names(found) == ["Running Shoes", "Leather Boot"]


def test_find_by_ids_keeps_order(session):
    assert [p.product_id for p in ProductRepository(session).find_by_ids([3, 1])] == [3, 1]


def test_trending_counts_and_is_distinct(session):
    repo = SearchHistoryRepository(session)
    for q in ["laptop", "phone", "laptop"]:
        repo.save(SearchHistory(customer_id=1, search_type="KEYWORD", query_text=q))
    assert ProductRepository(session).get_trending_keywords(5) == ["laptop", "phone"]


def test_recent_is_distinct_latest_first(session):
    repo = SearchHistoryRepository(session)
    for q in ["laptop", "phone", "laptop", "watch"]:
        repo.save(SearchHistory(customer_id=1, search_type="KEYWORD", query_text=q))
    assert repo.get_recent_searches(1, 5) == ["watch", "laptop", "phone"]


def test_save_image_history_with_detail(session):
    repo = SearchHistoryRepository(session)
    h = repo.save(SearchHistory(customer_id=1, search_type="IMAGE", image_url="/u/x.png"),
                  ImageSearchQuery(image_path="data/uploads/x.png", extracted_features="[0.1]"))
    detail = session.query(ImageSearchQuery).one()
    assert detail.search_id == h.search_id
    assert detail.image_path == "data/uploads/x.png"
