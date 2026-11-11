"""Home page catalogue: categories and paged product listing."""
from app.domain.models import Category, Product, SearchHistory
from tests.test_api import make_client  # noqa: F401  (fixture)


def _add_more(factory):
    with factory() as s:
        cat = s.get(Category, 2)  # mens-shoes from the fixture
        for i in range(3, 8):
            s.add(Product(product_id=i, name=f"Shoe {i}", description="", price=10 + i,
                          stock=1, image_url=f"/images/{i}.jpg", category=cat))
        s.commit()


def test_categories_with_counts_sorted_by_name(make_client):
    client, factory = make_client()
    _add_more(factory)
    assert client.get("/api/categories").json() == [
        {"name": "laptops", "count": 1}, {"name": "mens-shoes", "count": 6}]


def test_products_listing_is_paged(make_client):
    client, factory = make_client()
    _add_more(factory)
    first = client.get("/api/products", params={"limit": 4}).json()
    assert first["total"] == 7
    assert [p["id"] for p in first["products"]] == [1, 2, 3, 4]
    second = client.get("/api/products", params={"limit": 4, "offset": 4}).json()
    assert [p["id"] for p in second["products"]] == [5, 6, 7]


def test_products_filtered_by_category(make_client):
    client, factory = make_client()
    _add_more(factory)
    body = client.get("/api/products", params={"category": "laptops"}).json()
    assert body["total"] == 1
    assert [p["name"] for p in body["products"]] == ["Gaming Laptop"]


def test_unknown_category_is_empty_not_error(make_client):
    client, _ = make_client()
    assert client.get("/api/products", params={"category": "nope"}).json() == {
        "products": [], "total": 0}


def test_limit_is_capped(make_client):
    client, _ = make_client()
    assert client.get("/api/products", params={"limit": 1000}).status_code == 422


def test_browsing_does_not_write_search_history(make_client):
    client, factory = make_client()
    client.get("/api/products")
    with factory() as s:
        assert s.query(SearchHistory).count() == 0
