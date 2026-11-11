# Multimodal Product Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local FastAPI prototype that implements Search by Keyword, Search by Voice and Search by Image exactly as the report's design (Figures 4.3–4.7).

**Architecture:** Four packages mirroring Figure 4.3 — `presentation` (static HTML/JS), `control` (controller + services + AI wrappers), `data_access` (SQLAlchemy repositories + numpy `VectorIndex`), `domain` (10 entities). AI models sit behind small protocols (`TextEmbedder`, `ImageEmbedder`, `SpeechRecognizer`, `ObjectDetector`) so tests inject fakes and never download models.

**Tech Stack:** Python 3.14, FastAPI, Uvicorn, SQLAlchemy 2, SQLite, numpy, Pillow, sentence-transformers (CLIP), openai-whisper, ultralytics (YOLOv8n), pytest, httpx.

**Spec:** `docs/superpowers/specs/2026-11-12-multimodal-search-design.md`

## Global Constraints

- Keyword valid = 2–50 characters after `strip()`; message exactly `Keyword must be 2-50 characters`.
- Image valid = JPEG or PNG by magic bytes, size ≤ 10 * 1024 * 1024 bytes; message exactly `Only JPG/PNG, max 10MB`.
- Unrecognised speech message exactly `Please speak again`; rule: stripped text empty OR `no_speech_prob > 0.6`.
- Model-unavailable message exactly `Voice/Image search is unavailable, please type a keyword` (HTTP 503).
- Unknown upload id message exactly `Upload expired, please try again` (HTTP 404).
- Empty results: HTTP 200 with `{"products": []}`; UI shows `No products found` (keyword/voice) or `No matching products` (image).
- Result limits: keyword/voice ≤ 24, image ≤ 12. YOLO confidence ≥ 0.35, ≤ 8 objects. Preprocess long side ≤ 1024.
- History is saved only when results are non-empty; `search_type` ∈ `KEYWORD`, `VOICE`, `IMAGE`; demo customer id = 1.
- Money columns `Numeric(10, 2)`, timestamps `DateTime`. Vectors L2-normalised float32, dim 512.
- Class names as in Figure 4.3; methods snake_case with the diagram name in the docstring.
- No git commits unless the user asks (the user has not asked) — steps end at "tests pass".

## Review Focus

- Vietnamese input with diacritics and mixed case ("Giày  CHẠY bộ") must match the same products as "giay chay bo" — test in Task 2 and Task 6.
- A file renamed `.png` that is really a GIF/text must be rejected — test in Task 7.
- Searching while the vector index is empty (seed not run) must return keyword matches only, not crash — test in Task 6.
- A box outside or partly outside the image (or zero-area) must be clamped or fall back to the full image, not raise — test in Task 7.
- Two searches for the same keyword must create two history rows but trending must count it once per occurrence and return distinct keywords — test in Task 4.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `requirements.txt`, `requirements-ai.txt` | Core deps (tests run with these) / heavy AI deps |
| `app/config.py` | Paths, limits, thresholds (`SEMANTIC_MIN = 0.22`, `IMAGE_MIN = 0.55` initial) |
| `app/domain/models.py` | 10 SQLAlchemy entities |
| `app/data_access/database.py` | `make_engine(url)`, `make_session_factory(engine)`, `init_db(engine)` |
| `app/data_access/product_repository.py` | `ProductRepository` |
| `app/data_access/search_history_repository.py` | `SearchHistoryRepository` |
| `app/data_access/vector_index.py` | `VectorIndex` |
| `app/control/text_utils.py` | `normalize_keyword`, `validate_keyword` |
| `app/control/embedding.py` | protocols + `ClipTextEmbedder`, `ClipImageEmbedder` |
| `app/control/search_service.py` | `SearchService`, `ProductResult` |
| `app/control/speech_to_text_service.py` | `SpeechToTextService`, `WhisperRecognizer` |
| `app/control/image_analysis_service.py` | `ImageAnalysisService`, `YoloDetector`, `validate_image_file` |
| `app/control/upload_store.py` | `UploadStore` (save/load by UUID) |
| `app/control/search_controller.py` | `SearchController` router + `AppServices` container |
| `app/main.py` | `create_app(services=None)` |
| `app/presentation/static/*` | `index.html`, `styles.css`, `search_ui.js`, `voice_search_ui.js`, `image_search_ui.js`, `permission_manager.js` |
| `scripts/seed.py` | DummyJSON → SQLite + images + `vectors.npz` |
| `tests/conftest.py`, `tests/fakes.py` | temp DB fixture, fake models |

---

### Task 1: Project scaffold, config, domain models, database

**Files:**
- Create: `requirements.txt` (fastapi, uvicorn[standard], sqlalchemy>=2, numpy, pillow, python-multipart, httpx, pytest, requests), `requirements-ai.txt` (torch, sentence-transformers, openai-whisper, ultralytics), `app/__init__.py` and package `__init__.py` files, `app/config.py`, `app/domain/models.py`, `app/data_access/database.py`, `tests/conftest.py`, `.gitignore` (`data/`, `__pycache__/`, `.venv/`)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: models `Customer, Category, Product, Cart, CartItem, Order, OrderItem, SearchHistory, VoiceSearchQuery, ImageSearchQuery` with table/column names of spec §4 (e.g. `Product.__tablename__ == "products"`, columns `product_id, category_id, name, description, price, image_url, stock`; `SearchHistory`: `search_id, customer_id, search_type, query_text, image_url, created_at`; `VoiceSearchQuery`: `voice_query_id, search_id (unique), audio_path, recognized_text`; `ImageSearchQuery`: `image_query_id, search_id (unique), image_path, extracted_features`). `Product.category` relationship.
- Produces: `make_engine(url: str) -> Engine`, `make_session_factory(engine) -> sessionmaker`, `init_db(engine) -> None`.
- Produces fixture `session_factory` in `tests/conftest.py` (temp-file SQLite, tables created, Customer id 1 inserted).
- `config.py`: `DATA_DIR`, `DB_URL`, `IMAGES_DIR`, `UPLOADS_DIR`, `VECTORS_PATH`, `MAX_IMAGE_BYTES = 10*1024*1024`, `KEYWORD_MIN = 2`, `KEYWORD_MAX = 50`, `KEYWORD_LIMIT = 24`, `IMAGE_LIMIT = 12`, `SEMANTIC_MIN`, `IMAGE_MIN`, `NO_SPEECH_MAX = 0.6`, `YOLO_CONF = 0.35`, `YOLO_MAX = 8`, `MAX_SIDE = 1024`, `DEMO_CUSTOMER_ID = 1`.

- [ ] **Step 1:** Create venv and install core deps: `python -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`. Expected: install succeeds.
- [ ] **Step 2: Write failing tests** `test_all_ten_tables_created` (inspector table names == the 10 names of spec §4), `test_detail_search_id_unique` (inserting two `VoiceSearchQuery` rows with same `search_id` raises `IntegrityError`).
- [ ] **Step 3:** Run `.venv\Scripts\pytest tests/test_models.py -v` → FAIL (import error).
- [ ] **Step 4:** Implement config, models, database.
- [ ] **Step 5:** Run tests → PASS.

### Task 2: Keyword normalisation and validation

**Files:** Create `app/control/text_utils.py`; Test `tests/test_text_utils.py`

**Interfaces:**
- Produces: `normalize_keyword(text: str) -> str` (lower-case, remove Vietnamese diacritics incl. `đ→d`, collapse whitespace, strip). `validate_keyword(text: str) -> str` returns stripped text or raises `ValueError("Keyword must be 2-50 characters")`.

- [ ] **Step 1: Failing tests:** `normalize_keyword("Giày  CHẠY bộ") == "giay chay bo"`; `normalize_keyword("Đồng hồ") == "dong ho"`; `validate_keyword(" ab ") == "ab"`; `validate_keyword("a")`, `validate_keyword("   ")`, `validate_keyword("x"*51)` raise with exact message; `validate_keyword("x"*50)` ok.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement with `unicodedata.normalize("NFD")`, drop `Mn` chars, replace `đ/Đ` first.
- [ ] **Step 4:** Run → PASS.

### Task 3: VectorIndex

**Files:** Create `app/data_access/vector_index.py`; Test `tests/test_vector_index.py`

**Interfaces:**
- Produces: `class VectorIndex` with `__init__(product_ids: list[int], image_vecs: np.ndarray, text_vecs: np.ndarray)`, `@classmethod load(path) -> VectorIndex` (missing file → empty index), `save(path)`, `is_empty -> bool`, `nearest(query: np.ndarray, k: int, min_score: float, space: Literal["image","text","both"]) -> list[tuple[int, float]]` sorted by score desc. `"both"` = max of the two cosine scores per product. Query is normalised inside.

- [ ] **Step 1: Failing tests:** 3 products with orthogonal vectors → `nearest` returns the matching id first; `min_score` filters out others; `k=1` returns 1; empty index returns `[]`; save→load round-trip equals.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (matrix @ query). **Step 4:** Run → PASS.

### Task 4: Repositories

**Files:** Create `app/data_access/product_repository.py`, `app/data_access/search_history_repository.py`; Test `tests/test_repositories.py`

**Interfaces:**
- Consumes: models (Task 1), `normalize_keyword` (Task 2).
- Produces `ProductRepository(session)`:
  - `find_by_keyword(normalized: str, limit: int) -> list[Product]` — products whose normalised name, category name or description contains `normalized`; name matches before category before description matches; Python-side normalisation over all products is acceptable (≤ 200 rows).
  - `find_by_ids(ids: list[int]) -> list[Product]` preserving input order.
  - `get_trending_keywords(limit: int = 5) -> list[str]` — most frequent `query_text` of KEYWORD/VOICE histories (all customers), distinct, ties by latest.
- Produces `SearchHistoryRepository(session)`:
  - `get_recent_searches(customer_id: int, limit: int = 5) -> list[str]` — distinct latest `query_text` of KEYWORD/VOICE.
  - `save(history: SearchHistory, detail: VoiceSearchQuery | ImageSearchQuery | None = None) -> SearchHistory` — commits, sets detail.search_id.

- [ ] **Step 1: Failing tests:** seeded products "Running Shoes" (category "mens-shoes"), "Gaming Laptop", "Shoe Rack" (description "holds shoes"): `find_by_keyword("shoe")` → Running Shoes and Shoe Rack before any description-only match; `find_by_ids([3,1])` keeps order; two saves of "laptop" + one of "phone" → trending `["laptop","phone"]`; recent returns distinct latest first; saving IMAGE with detail stores `image_path`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS.

### Task 5: Embedding protocols and fakes

**Files:** Create `app/control/embedding.py`, `tests/fakes.py`; Test `tests/test_embedding_fakes.py`

**Interfaces:**
- Produces protocols: `TextEmbedder.encode_text(texts: list[str]) -> np.ndarray` (n×512, L2-normalised); `ImageEmbedder.encode_image(images: list[PIL.Image.Image]) -> np.ndarray`.
- Produces `ClipTextEmbedder(device)` using `SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")`, `ClipImageEmbedder(device)` using `SentenceTransformer("clip-ViT-B-32")`; imports of torch/sentence-transformers inside `__init__` so core tests do not need them. `pick_device() -> str` ("cuda" if available else "cpu").
- Produces in `tests/fakes.py`: `FakeTextEmbedder(mapping: dict[str, np.ndarray], default)` (looks up normalised text), `FakeImageEmbedder(vec)`, `FakeRecognizer(text, no_speech_prob, language="vi")`, `FakeDetector(objects)`.

- [ ] **Step 1:** Failing test: fakes return arrays of shape (n, 512) with unit norm. **Step 2–4:** FAIL → implement → PASS.

### Task 6: SearchService

**Files:** Create `app/control/search_service.py`; Test `tests/test_search_service.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces `@dataclass ProductResult(id, name, description, price: float, image_url, stock, category, score: float)`.
- Produces `SearchService(products: ProductRepository, history: SearchHistoryRepository, index: VectorIndex, text_embedder: TextEmbedder | None, image_embedder: ImageEmbedder | None)`:
  - `find_by_keyword(keyword: str) -> list[ProductResult]` — validate, normalise, set A from repository (score 1.0), set B from `index.nearest(text_vec, k=KEYWORD_LIMIT, min_score=SEMANTIC_MIN, space="both")` using the **raw** stripped keyword for the embedder; merge A then B, dedupe, cap 24. Embedder `None`, raising, or empty index → A only.
  - `find_similar_products(features: np.ndarray) -> list[ProductResult]` — `space="image"`, `IMAGE_MIN`, cap 12.
  - `save_search_history(customer_id: int, search_type: str, query_text: str | None, image_url: str | None = None, detail=None) -> None`.
  - `normalize_keyword` reused from Task 2 (diagram's private method).

- [ ] **Step 1: Failing tests:** keyword exact match first; Vietnamese "giày chạy bộ" via fake embedder mapped to Running Shoes vector returns Running Shoes; "Giày  CHẠY bộ" returns the same; empty index → keyword matches only, no error; embedder raising → keyword matches only; nonsense keyword → `[]`; invalid keyword raises `ValueError`; image features closest → that product first, below `IMAGE_MIN` excluded.
- [ ] **Step 2–4:** FAIL → implement → PASS.

### Task 7: SpeechToTextService, ImageAnalysisService, UploadStore

**Files:** Create `app/control/speech_to_text_service.py`, `app/control/image_analysis_service.py`, `app/control/upload_store.py`; Test `tests/test_ai_services.py`

**Interfaces:**
- Produces `UploadStore(dir: Path)`: `save(data: bytes, suffix: str) -> str` (uuid id), `path(id) -> Path` raising `KeyError` if missing; ids must match `^[0-9a-f]{32}$` (reject path traversal).
- Produces `class SpeechNotRecognized(Exception)`, protocol `SpeechRecognizer.transcribe(path: Path) -> dict(text, language, no_speech_prob)`; `WhisperRecognizer(model="small", device)` (lazy import whisper; `no_speech_prob` = max over segments; `fp16` only on cuda).
- Produces `SpeechToTextService(recognizer)`: `convert_speech_to_text(audio_path: Path) -> tuple[str, str]` (text, language) raising `SpeechNotRecognized` per Global Constraints. (noiseReduction/detectLanguage are inside Whisper — say so in docstring.)
- Produces `validate_image_file(data: bytes) -> str` returning `"jpg"`/`"png"` or raising `ValueError("Only JPG/PNG, max 10MB")`.
- Produces `@dataclass DetectedObject(label: str, confidence: float, box: tuple[int,int,int,int])`, protocol `ObjectDetector.detect(img) -> list[DetectedObject]`, `YoloDetector(device)` (lazy import ultralytics, `yolov8n.pt`).
- Produces `ImageAnalysisService(detector, image_embedder)`: `preprocess_image(data: bytes) -> PIL.Image` (EXIF transpose, RGB, long side ≤ 1024); `detect_objects(img) -> list[DetectedObject]` (conf ≥ 0.35, ≤ 8, sorted by confidence); `extract_features(img, box: tuple|None) -> np.ndarray` (clamp box to image; zero-area or None → full image).

- [ ] **Step 1: Failing tests:** valid PNG/JPEG bytes (made with Pillow) accepted; GIF bytes renamed rejected; 10 MB + 1 rejected; `UploadStore.path("../x")` raises `KeyError`; fake recognizer `""` → `SpeechNotRecognized`; `no_speech_prob=0.7` → raises; `0.2` with "điện thoại" → returns text; preprocess 3000×1000 → 1024×341; box (-50,-50,99999,99999) clamps without error; box (10,10,10,10) uses full image; detector results with conf 0.2 filtered.
- [ ] **Step 2–4:** FAIL → implement → PASS.

### Task 8: SearchController API and app factory

**Files:** Create `app/control/search_controller.py`, `app/main.py`; Test `tests/test_api.py`

**Interfaces:**
- Consumes: Tasks 4–7.
- Produces `@dataclass AppServices(session_factory, index, text_embedder, image_embedder, recognizer, detector, uploads: UploadStore)` and `build_real_services() -> AppServices` (loads models lazily; a model that fails to load is set to `None` and logged).
- Produces `create_app(services: AppServices | None = None) -> FastAPI` — mounts `app/presentation/static` at `/`, `data/images` at `/images`, router at `/api`.
- Endpoints and responses exactly as spec §5 (including `GET /api/products/{id}` → ProductResult with score 0, 404 if missing), errors as spec §7 (JSON `{"detail": "<message>"}`). Voice/image endpoints return 503 when the needed model is `None`. History saving per spec §6 with `DEMO_CUSTOMER_ID`.

- [ ] **Step 1: Failing tests (TestClient + fakes):** suggestions empty then populated after a keyword search; keyword `q=a` → 400 + exact message; `q=laptop` → 200 with Gaming Laptop and one history row; nonsense → 200 `[]` and no history row; transcribe with fake text → `{audio_id, text, language}`; fake empty text → 422 `Please speak again`; `/search/voice` unknown audio_id → 404; valid → VOICE history + VoiceSearchQuery row; analyze GIF → 400; analyze PNG → objects list + image_id; `/search/image` with box → IMAGE history + ImageSearchQuery row; recognizer `None` → 503 with exact message.
- [ ] **Step 2–4:** FAIL → implement → PASS. Then run full suite `.venv\Scripts\pytest -q` → all PASS.

### Task 9: Seed script

**Files:** Create `scripts/seed.py`; Test `tests/test_seed.py`

**Interfaces:**
- Produces `map_dummyjson_product(raw: dict) -> dict` (keys `name, description, price, stock, category, thumbnail_url, product_id`), `product_text(p) -> str` (`f"{name}. {category}. {description}"`), `main(limit: int | None = None, skip_vectors: bool = False)`. Source `https://dummyjson.com/products?limit=0`; images to `data/images/{id}.jpg`; `image_url = "/images/{id}.jpg"`; categories created from distinct `category`; vectors via real CLIP embedders batched 32; writes `VECTORS_PATH`. Re-run drops and recreates DB, images reused if present.

- [ ] **Step 1:** Failing test on `map_dummyjson_product` with a literal sample dict (title→name, price float, category slug kept).
- [ ] **Step 2–4:** FAIL → implement → PASS.

### Task 10: Frontend (Presentation layer)

**Files:** Create `app/presentation/static/index.html`, `styles.css`, `search_ui.js`, `voice_search_ui.js`, `image_search_ui.js`, `permission_manager.js`

**Interfaces:**
- Consumes: spec §5 endpoints. Each JS file exports one class named as in Figure 4.3 with the diagram's methods (camelCase), e.g. `SearchUI.selectSearchBar()`, `submitKeyword()`, `validateFormat()`, `showSuggestedKeywords()`, `displayProducts()`, `showNoProductsFound()`; `VoiceSearchUI` uses `MediaRecorder` (webm) with a recording indicator, shows editable transcript + Confirm / Speak again; `ImageSearchUI` shows upload popup, preview on `<canvas>` with detected boxes (click to select) and drag-to-draw a custom box, Confirm selection; `PermissionManager.checkMicPermission()/checkCameraPermission()/requestPermission()` via `navigator.permissions` + `getUserMedia` (camera button also allows file upload without camera).
- Layout: header search bar with mic and camera icons; suggestions dropdown (Recent / Trending); result grid of cards (image, name, category, price, stock); product detail modal on click.

- [ ] **Step 1:** Implement files; validation messages exactly as Global Constraints.
- [ ] **Step 2:** Run app with fakes-free core (`uvicorn app.main:app`) after Task 9 seed with `--skip-vectors`; open in browser pane; keyword search works and invalid keyword shows message. Expected: no console errors.

### Task 11: Real models, seed, smoke test, README

**Files:** Create `README.md`; Modify `app/config.py` thresholds if tuning needs it.

- [ ] **Step 1:** Install AI deps: `.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128` (fall back to CPU wheel if no CUDA build for 3.14), then `-r requirements-ai.txt`. Expected: `python -c "import torch;print(torch.cuda.is_available())"` prints True (or False with CPU fallback noted).
- [ ] **Step 2:** `.venv\Scripts\python scripts/seed.py` → prints product count (~190) and writes `data/vectors.npz`.
- [ ] **Step 3:** Run server; smoke test: "laptop", "giày chạy bộ", invalid "a", voice "điện thoại" (user speaks, confirms), image of shoes (from `data/images` of a shoe product, plus a crop). Adjust `SEMANTIC_MIN`/`IMAGE_MIN` so nonsense ("qwxz") returns nothing and the four queries return relevant items.
- [ ] **Step 4:** Take screenshots of each flow for the report.
- [ ] **Step 5:** README: setup, seed, run, test commands; design-to-code mapping table (Figure 4.3 class → file); deviation note (SQLite + numpy instead of PostgreSQL + pgvector).
- [ ] **Step 6:** Full test suite PASS.
