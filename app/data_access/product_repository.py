"""ProductRepository (entity access): reads the product catalogue."""
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.control.text_utils import normalize_keyword
from app.domain.models import Category, Product, SearchHistory


class ProductRepository:
    def __init__(self, session: Session):
        self.session = session

    def _all(self) -> list[Product]:
        stmt = select(Product).options(selectinload(Product.category)).order_by(Product.product_id)
        return list(self.session.scalars(stmt))

    def find_by_keyword(self, normalized: str, limit: int,
                        include_description: bool = True) -> list[Product]:
        """findByKeyword(keyword): name matches first, then category, then description."""
        tiers: tuple[list[Product], list[Product], list[Product]] = ([], [], [])
        for p in self._all():
            if normalized in normalize_keyword(p.name):
                tiers[0].append(p)
            elif p.category and normalized in normalize_keyword(p.category.name.replace("-", " ")):
                tiers[1].append(p)
            elif include_description and normalized in normalize_keyword(p.description or ""):
                tiers[2].append(p)
        return (tiers[0] + tiers[1] + tiers[2])[:limit]

    def find_by_ids(self, ids: list[int]) -> list[Product]:
        if not ids:
            return []
        stmt = select(Product).options(selectinload(Product.category)).where(
            Product.product_id.in_(ids))
        by_id = {p.product_id: p for p in self.session.scalars(stmt)}
        return [by_id[i] for i in ids if i in by_id]

    def list_categories(self) -> list[tuple[str, int]]:
        """(category name, product count), by name — for the home page filter."""
        stmt = (select(Category.name, func.count(Product.product_id))
                .join(Product, Product.category_id == Category.category_id)
                .group_by(Category.name).order_by(Category.name))
        return [(name, count) for name, count in self.session.execute(stmt)]

    def list_products(self, category: str | None, offset: int,
                      limit: int) -> tuple[list[Product], int]:
        """One page of the catalogue (optionally one category) and the total count."""
        stmt = select(Product).options(selectinload(Product.category))
        count = select(func.count(Product.product_id))
        if category:
            stmt = stmt.join(Category).where(Category.name == category)
            count = count.join(Category).where(Category.name == category)
        page = self.session.scalars(stmt.order_by(Product.product_id).offset(offset).limit(limit))
        return list(page), self.session.scalar(count)

    def find_by_id(self, product_id: int) -> Product | None:
        found = self.find_by_ids([product_id])
        return found[0] if found else None

    def get_trending_keywords(self, limit: int = 5) -> list[str]:
        """getTrendingKeywords(): most searched keyword/voice texts across all customers."""
        stmt = (
            select(SearchHistory.query_text)
            .where(SearchHistory.search_type.in_(["KEYWORD", "VOICE"]),
                   SearchHistory.query_text.is_not(None))
            .order_by(SearchHistory.search_id.desc())
            .limit(2000)
        )
        # group ignoring case, accents and spaces; show the latest spelling;
        # ties go to the keyword searched most recently
        counts: dict[str, list] = {}
        for rank, text in enumerate(self.session.scalars(stmt)):
            entry = counts.setdefault(normalize_keyword(text), [0, rank, text])
            entry[0] += 1
        ordered = sorted(counts.values(), key=lambda e: (-e[0], e[1]))
        return [text for _, _, text in ordered[:limit]]
