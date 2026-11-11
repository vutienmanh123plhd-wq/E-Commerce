"""SearchController (control): HTTP entry point for the three search use cases.

Endpoint                      Design operation (Figure 4.3)
GET  /api/suggestions          getRecentSearches + getTrendingKeywords
GET  /api/search/keyword       searchByKeyword(keyword)
POST /api/voice/transcribe     convertSpeechToText(audio)
POST /api/search/voice         submitQuery(text)
POST /api/image/analyze        extractFeatures(image) -> preprocessImage, detectObjects
POST /api/search/image         searchSimilar(features)
GET  /api/products/{id}        View Product
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app import config
from app.control.image_analysis_service import ImageAnalysisService, validate_image_file
from app.control.search_service import ProductResult, SearchService
from app.control.speech_to_text_service import SpeechNotRecognized, SpeechToTextService
from app.control.upload_store import UploadStore
from app.data_access.product_repository import ProductRepository
from app.data_access.search_history_repository import SearchHistoryRepository
from app.data_access.vector_index import VectorIndex
from app.domain.models import ImageSearchQuery, VoiceSearchQuery

log = logging.getLogger(__name__)


@dataclass
class AppServices:
    session_factory: object
    index: VectorIndex
    text_embedder: object | None
    image_embedder: object | None
    recognizer: object | None
    detector: object | None
    uploads: UploadStore = field(default_factory=lambda: UploadStore(config.UPLOADS_DIR))
    translator: object | None = None
    semantic_min: float = config.SEMANTIC_MIN_Z


def build_real_services() -> AppServices:
    """Load the database, vector index and AI models; a model that fails becomes None."""
    from app.control.embedding import ClipImageEmbedder, ClipTextEmbedder, pick_device
    from app.control.image_analysis_service import YoloDetector
    from app.control.speech_to_text_service import WhisperRecognizer
    from app.control.translation import MarianTranslator
    from app.data_access.database import init_db, make_engine, make_session_factory

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = make_engine(config.DB_URL)
    init_db(engine)
    device = pick_device()
    log.info("loading AI models on %s", device)

    def load(name, factory):
        try:
            return factory()
        except Exception:
            log.exception("could not load %s; that search mode is disabled", name)
            return None

    return AppServices(
        session_factory=make_session_factory(engine),
        index=VectorIndex.load(config.VECTORS_PATH),
        text_embedder=load("text encoder", lambda: ClipTextEmbedder(device)),
        image_embedder=load("image encoder", lambda: ClipImageEmbedder(device)),
        recognizer=load("Whisper", lambda: WhisperRecognizer(device=device)),
        detector=load("YOLO", lambda: YoloDetector(device)),
        translator=load("vi-en translator", lambda: MarianTranslator(device)),
    )


class VoiceQuery(BaseModel):
    text: str
    audio_id: str


class ImageQuery(BaseModel):
    image_id: str
    box: list[int] | None = None


def _products(results: list[ProductResult]) -> dict:
    return {"products": [r.to_dict() for r in results]}


def build_router(services: AppServices) -> APIRouter:
    router = APIRouter(prefix="/api")

    def search_service(session) -> SearchService:
        return SearchService(ProductRepository(session), SearchHistoryRepository(session),
                             services.index, services.text_embedder, services.image_embedder,
                             translator=services.translator, semantic_min=services.semantic_min)

    def require(model):
        if model is None:
            raise HTTPException(503, config.MSG_MODEL_UNAVAILABLE)
        return model

    def upload_path(upload_id: str) -> Path:
        try:
            return services.uploads.path(upload_id)
        except KeyError:
            raise HTTPException(404, config.MSG_UPLOAD_EXPIRED)

    def read_limited(upload: UploadFile, limit: int, message: str) -> bytes:
        data = upload.file.read(limit + 1)  # never load more than the limit into memory
        if len(data) > limit:
            raise HTTPException(400, message)
        return data

    def tidy(text: str) -> str:
        return " ".join(text.split())

    def search_text(session, text: str) -> list[ProductResult]:
        try:
            return search_service(session).find_by_keyword(text)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @router.get("/suggestions")
    def suggestions():
        with services.session_factory() as s:
            return {
                "recent": SearchHistoryRepository(s).get_recent_searches(config.DEMO_CUSTOMER_ID, 5),
                "trending": ProductRepository(s).get_trending_keywords(5),
            }

    @router.get("/search/keyword")
    def search_by_keyword(q: str = ""):
        with services.session_factory() as s:
            results = search_text(s, q)
            if results:
                search_service(s).save_search_history(config.DEMO_CUSTOMER_ID, "KEYWORD",
                                                      tidy(q))
            return _products(results)

    @router.post("/voice/transcribe")
    def convert_speech_to_text(audio: UploadFile = File(...)):
        recognizer = require(services.recognizer)
        suffix = Path(audio.filename or "").suffix.lstrip(".").lower()
        if suffix not in config.AUDIO_SUFFIXES:  # never store a client-chosen extension
            suffix = "webm"
        data = read_limited(audio, config.MAX_AUDIO_BYTES, config.MSG_AUDIO_TOO_LONG)
        audio_id = services.uploads.save(data, suffix)
        try:
            text, language = SpeechToTextService(recognizer).convert_speech_to_text(
                services.uploads.path(audio_id))
        except SpeechNotRecognized:
            raise HTTPException(422, config.MSG_SPEAK_AGAIN)
        return {"audio_id": audio_id, "text": text, "language": language}

    @router.post("/search/voice")
    def submit_query(query: VoiceQuery):
        path = upload_path(query.audio_id)
        with services.session_factory() as s:
            results = search_text(s, query.text)
            if results:
                search_service(s).save_search_history(
                    config.DEMO_CUSTOMER_ID, "VOICE", tidy(query.text),
                    detail=VoiceSearchQuery(audio_path=str(path),
                                            recognized_text=tidy(query.text)))
            return _products(results)

    def image_service() -> ImageAnalysisService:
        return ImageAnalysisService(services.detector, require(services.image_embedder))

    @router.post("/image/analyze")
    def extract_features(image: UploadFile = File(...)):
        analysis = image_service()
        data = read_limited(image, config.MAX_IMAGE_BYTES, config.MSG_INVALID_IMAGE)
        try:
            suffix = validate_image_file(data)
            img = analysis.preprocess_image(data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        image_id = services.uploads.save(data, suffix)
        objects = analysis.detect_objects(img)
        return {
            "image_id": image_id, "width": img.width, "height": img.height,
            "objects": [{"label": o.label, "confidence": round(o.confidence, 3),
                         "box": list(o.box)} for o in objects],
        }

    @router.post("/search/image")
    def search_similar(query: ImageQuery):
        analysis = image_service()
        path = upload_path(query.image_id)
        try:
            img = analysis.preprocess_image(path.read_bytes())
        except ValueError:  # e.g. an audio id sent as image_id
            raise HTTPException(404, config.MSG_UPLOAD_EXPIRED)
        features = analysis.extract_features(img, query.box)
        with services.session_factory() as s:
            service = search_service(s)
            results = service.find_similar_products(features)
            if results:
                service.save_search_history(
                    config.DEMO_CUSTOMER_ID, "IMAGE", None, image_url=f"/uploads/{path.name}",
                    detail=ImageSearchQuery(image_path=str(path),
                                            extracted_features=json.dumps(
                                                [round(float(v), 5) for v in features])))
            return _products(results)

    @router.get("/categories")
    def categories():
        with services.session_factory() as s:
            return [{"name": n, "count": c} for n, c in ProductRepository(s).list_categories()]

    @router.get("/products")
    def browse_products(category: str | None = None, offset: int = Query(0, ge=0),
                        limit: int = Query(24, ge=1, le=100)):
        """Home page catalogue (browsing is not a search, so no history is written)."""
        with services.session_factory() as s:
            products, total = ProductRepository(s).list_products(category, offset, limit)
            return {"products": [ProductResult.from_product(p, 0.0).to_dict() for p in products],
                    "total": total}

    @router.get("/products/{product_id}")
    def view_product(product_id: int):
        with services.session_factory() as s:
            product = ProductRepository(s).find_by_id(product_id)
            if product is None:
                raise HTTPException(404, "Product not found")
            return ProductResult.from_product(product, 0.0).to_dict()

    return router
