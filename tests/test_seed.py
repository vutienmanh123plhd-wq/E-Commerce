from scripts.seed import map_dummyjson_product, product_text

RAW = {
    "id": 7, "title": "Nike Air Jordan 1 Red And Black", "description": "Iconic sneakers.",
    "category": "mens-shoes", "price": 149.99, "stock": 7,
    "thumbnail": "https://cdn.dummyjson.com/x/thumbnail.webp", "rating": 4.8,
}


def test_map_dummyjson_product():
    assert map_dummyjson_product(RAW) == {
        "product_id": 7, "name": "Nike Air Jordan 1 Red And Black",
        "description": "Iconic sneakers.", "price": 149.99, "stock": 7,
        "category": "mens-shoes", "thumbnail_url": "https://cdn.dummyjson.com/x/thumbnail.webp",
    }


def test_product_text_is_clip_prompt_with_name_and_category():
    p = map_dummyjson_product(RAW)
    assert product_text(p) == "a photo of Nike Air Jordan 1 Red And Black, mens shoes"
