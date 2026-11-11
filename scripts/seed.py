"""Seed the demo catalogue from DummyJSON and build the product vector index.

Usage:  python -m scripts.seed [--limit N] [--skip-vectors]
"""
import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402

from app import config  # noqa: E402
from app.data_access.database import init_db, make_engine, make_session_factory  # noqa: E402
from app.data_access.vector_index import VectorIndex  # noqa: E402
from app.domain.models import Base, Category, Customer, Product  # noqa: E402

SOURCE = "https://dummyjson.com/products?limit=0"


def map_dummyjson_product(raw: dict) -> dict:
    return {
        "product_id": int(raw["id"]), "name": raw["title"], "description": raw.get("description", ""),
        "price": float(raw["price"]), "stock": int(raw.get("stock", 0)),
        "category": raw["category"], "thumbnail_url": raw["thumbnail"],
    }


def product_text(p: dict) -> str:
    """CLIP prompt for the product's text vector (encoded with the English CLIP text encoder)."""
    return f"a photo of {p['name']}, {p['category'].replace('-', ' ')}"


def download_image(p: dict) -> Path:
    path = config.IMAGES_DIR / f"{p['product_id']}.jpg"
    if not path.exists():
        data = requests.get(p["thumbnail_url"], timeout=30).content
        Image.open(io.BytesIO(data)).convert("RGB").save(path, "JPEG", quality=90)
    return path


def main(limit: int | None = None, skip_vectors: bool = False) -> None:
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading catalogue ...")
    raw = requests.get(SOURCE, timeout=60).json()["products"]
    items = [map_dummyjson_product(r) for r in raw][:limit]

    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(download_image, items))
    print(f"Images ready: {len(paths)}")

    engine = make_engine(config.DB_URL)
    Base.metadata.drop_all(engine)
    init_db(engine)
    with make_session_factory(engine)() as s:
        s.add(Customer(customer_id=config.DEMO_CUSTOMER_ID, full_name="Demo Customer",
                       email="demo@example.com", password="not-used", phone="0900000000",
                       address="Hanoi"))
        cats = {name: Category(name=name) for name in sorted({p["category"] for p in items})}
        s.add_all(cats.values())
        for p in items:
            s.add(Product(product_id=p["product_id"], name=p["name"], description=p["description"],
                          price=p["price"], stock=p["stock"], category=cats[p["category"]],
                          image_url=f"/images/{p['product_id']}.jpg"))
        s.commit()
    engine.dispose()
    print(f"Products saved: {len(items)} in {len(cats)} categories")

    if skip_vectors:
        print("Skipped vectors (--skip-vectors): semantic and image search disabled.")
        return
    from app.control.embedding import ClipImageEmbedder, pick_device
    device = pick_device()
    print(f"Encoding on {device} ...")
    clip = ClipImageEmbedder(device)
    images = [Image.open(path).convert("RGB") for path in paths]
    image_vecs = clip.encode_image(images)
    # Product texts use CLIP's own English text encoder; customer queries use the
    # multilingual encoder, which was trained to land in this same space.
    text_vecs = clip.encode_text([product_text(p) for p in items])
    VectorIndex([p["product_id"] for p in items], image_vecs, text_vecs).save(config.VECTORS_PATH)
    print(f"Vector index saved: {config.VECTORS_PATH} {image_vecs.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-vectors", action="store_true")
    args = parser.parse_args()
    main(args.limit, args.skip_vectors)
