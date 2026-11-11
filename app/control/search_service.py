"""SearchService (control): ranks products for keyword, voice and image queries."""
import logging
from dataclasses import asdict, dataclass

import numpy as np

from app import config
from app.control.embedding import ImageEmbedder, TextEmbedder
from app.control.text_utils import normalize_keyword, validate_keyword
from app.control.translation import Translator, clean_translation, has_vietnamese
from app.data_access.product_repository import ProductRepository
from app.data_access.search_history_repository import SearchHistoryRepository
from app.data_access.vector_index import VectorIndex
from app.domain.models import ImageSearchQuery, Product, SearchHistory, VoiceSearchQuery

log = logging.getLogger(__name__)


@dataclass
class ProductResult:
    id: int
    name: str
    description: str
    price: float
    image_url: str | None
    stock: int
    category: str | None
    score: float

    @classmethod
    def from_product(cls, p: Product, score: float) -> "ProductResult":
        return cls(p.product_id, p.name, p.description or "", float(p.price), p.image_url,
                   p.stock, p.category.name if p.category else None, round(score, 4))

    def to_dict(self) -> dict:
        return asdict(self)


class SearchService:
    def __init__(self, products: ProductRepository, history: SearchHistoryRepository,
                 index: VectorIndex, text_embedder: TextEmbedder | None,
                 image_embedder: ImageEmbedder | None, translator: Translator | None = None,
                 semantic_min: float = config.SEMANTIC_MIN_Z):
        self.products = products
        self.history = history
        self.index = index
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.translator = translator
        self.semantic_min = semantic_min

    normalize_keyword = staticmethod(normalize_keyword)  # normalizeKeyword(keyword)

    def find_by_keyword(self, keyword: str) -> list[ProductResult]:
        """findByKeyword(keyword): exact keyword matches first, then semantic matches.

        A Vietnamese query is also translated to English; both variants are matched.
        """
        keyword = validate_keyword(keyword)
        variants = [keyword] + self._translations(keyword)
        results: list[ProductResult] = []
        seen: set[int] = set()

        def add(product: Product | None, score: float) -> None:
            if product is not None and product.product_id not in seen \
                    and len(results) < config.KEYWORD_LIMIT:
                results.append(ProductResult.from_product(product, score))
                seen.add(product.product_id)

        for i, variant in enumerate(variants):
            # a machine translation is less reliable: match it on names and categories only
            for p in self.products.find_by_keyword(self.normalize_keyword(variant),
                                                   config.KEYWORD_LIMIT,
                                                   include_description=i == 0):
                add(p, 1.0)
        for pid, z in self._semantic(variants):
            add(self.products.find_by_id(pid), min(z / 4, 0.99))
        return results

    def _translations(self, keyword: str) -> list[str]:
        if self.translator is None or not has_vietnamese(keyword):
            return []
        try:
            cleaned = clean_translation(keyword, self.translator.translate(keyword))
        except Exception:
            log.exception("translator failed; searching the original query only")
            return []
        return [cleaned] if cleaned else []

    def _semantic(self, variants: list[str]) -> list[tuple[int, float]]:
        if self.text_embedder is None or self.index.is_empty:
            return []
        try:
            queries = self.text_embedder.encode_text(variants)
        except Exception:  # keep keyword search alive (NFR04)
            log.exception("text encoder failed; keyword matches only")
            return []
        return self.index.fused_nearest(queries, config.KEYWORD_LIMIT, self.semantic_min)

    def find_similar_products(self, features: np.ndarray) -> list[ProductResult]:
        """findSimilarProducts(features): products whose image is closest to the features.

        A photo found on the web is a different picture of the product, so its cosine to
        the catalogue image is only 0.5-0.85. Products are therefore ranked by the fused
        z-score of image-to-image and image-to-text similarity (CLIP aligns images with
        text), like keyword search. Results are shown only if the best product passes
        the gate: fused z >= IMAGE_GATE_Z, or a near-duplicate (cosine >= IMAGE_GATE).
        The score shown to the Customer is the honest image cosine.
        """
        cosine = dict(self.index.nearest(features, len(self.index.product_ids), -1.0, "image"))
        fused = self.index.fused_nearest(np.atleast_2d(features), config.IMAGE_LIMIT, -np.inf)
        duplicates = [pid for pid, c in sorted(cosine.items(), key=lambda kv: -kv[1])
                      if c >= config.IMAGE_GATE]
        if not duplicates and (not fused or fused[0][1] < config.IMAGE_GATE_Z):
            return []
        related = [pid for pid, z in fused if z >= config.IMAGE_MIN_Z]
        ordered = list(dict.fromkeys(duplicates + related))[:config.IMAGE_LIMIT]
        by_id = {p.product_id: p for p in self.products.find_by_ids(ordered)}
        return [ProductResult.from_product(by_id[pid], cosine[pid]) for pid in ordered
                if pid in by_id]

    def save_search_history(self, customer_id: int, search_type: str, query_text: str | None,
                            image_url: str | None = None,
                            detail: VoiceSearchQuery | ImageSearchQuery | None = None) -> None:
        """saveSearchHistory(customerId, query, media)."""
        self.history.save(SearchHistory(customer_id=customer_id, search_type=search_type,
                                        query_text=query_text, image_url=image_url), detail)
