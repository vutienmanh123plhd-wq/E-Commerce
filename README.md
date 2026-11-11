# E-Commerce Multimodal Product Search

Prototype of the three search use cases from the report *E-Commerce Multimodal Product Search — System Analysis and Design*: **Search by Keyword**, **Search by Voice** and **Search by Image**.

## Requirements

- Windows, Python 3.14, about 6 GB free on the project drive (PyTorch + models).
- Chrome or Edge (microphone recording needs `localhost` or HTTPS).
- Optional: NVIDIA GPU with CUDA — models fall back to CPU without one.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements-ai.txt
```

Seed the catalogue (194 DummyJSON products, their images, and the vector index):

```bash
.venv\Scripts\python -m scripts.seed
```

Model weights (CLIP, multilingual CLIP, Whisper `small`, YOLOv8n) download on first use into `.models/`.

## Run

```bash
.venv\Scripts\uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

## Test

```bash
.venv\Scripts\python -m pytest -q
```

Tests use fake models, so they run without PyTorch or any download.

## Design-to-code mapping (Figure 4.3)

| Design class | Stereotype | File |
| --- | --- | --- |
| SearchUI | boundary | `app/presentation/static/search_ui.js` |
| VoiceSearchUI | boundary | `app/presentation/static/voice_search_ui.js` |
| ImageSearchUI | boundary | `app/presentation/static/image_search_ui.js` |
| PermissionManager | control | `app/presentation/static/permission_manager.js` (browser permissions) |
| SearchController | control | `app/control/search_controller.py` |
| SearchService | control | `app/control/search_service.py` |
| SpeechToTextService | control | `app/control/speech_to_text_service.py` (Whisper) |
| ImageAnalysisService | control | `app/control/image_analysis_service.py` (YOLOv8n + CLIP) |
| EmbeddingService (Figure 4.7) | control | `app/control/embedding.py` (CLIP, multilingual CLIP) |
| ProductRepository | entity access | `app/data_access/product_repository.py` |
| SearchHistoryRepository | entity access | `app/data_access/search_history_repository.py` |
| VectorIndex (Figure 4.7) | data | `app/data_access/vector_index.py` |
| 10 entities (Table 4.2) | entity | `app/domain/models.py` |

## API

| Endpoint | Design operation |
| --- | --- |
| `GET /api/suggestions` | getRecentSearches + getTrendingKeywords |
| `GET /api/search/keyword?q=` | searchByKeyword |
| `POST /api/voice/transcribe` | convertSpeechToText |
| `POST /api/search/voice` | submitQuery |
| `POST /api/image/analyze` | extractFeatures (preprocessImage, detectObjects) |
| `POST /api/search/image` | searchSimilar |
| `GET /api/products/{id}` | View Product |

## Deviations from the design

- **SQLite + numpy instead of PostgreSQL + pgvector.** The catalogue has under 200 products, so an in-memory cosine search takes well under a millisecond. Only `VectorIndex` would change to move to pgvector.
- **No login.** All searches belong to the demo customer (`customer_id = 1`); cart and order tables exist but have no screens.
- **PermissionManager runs in the browser,** because microphone and camera permissions belong to the browser.
- **noiseReduction() and detectLanguage()** are performed inside Whisper rather than as separate steps.
- **Vietnamese queries are also machine-translated** to English (Helsinki-NLP `opus-mt-vi-en`) and both forms are searched. The translation is matched against product names and categories only.

## How ranking works

- **Keyword / voice:** exact matches on the normalised keyword come first (name, then category, then description). Then come semantic matches, scored as `0.5·z(image cosine) + 0.5·z(text cosine)` over the whole catalogue and kept when the score is at least `SEMANTIC_MIN_Z = 1.65`. Raw cosine could not separate real queries from nonsense, because every text query scores about 0.83 against every product text.
- **Image:** products are ranked by the same fused z-score, comparing the photo with both product images and product texts. Results are shown when the best product reaches `IMAGE_GATE_Z = 2.1`, or when the photo is a near-duplicate of a catalogue picture (cosine ≥ `IMAGE_GATE = 0.84`). Related products down to `IMAGE_MIN_Z = 1.6` are listed. Tuned on 28 web photos: 22 return results and 21 have the right product type first, while blank, noise and gradient images return nothing.
- **Voice:** Whisper detects the language but may only choose Vietnamese or English. Letters from other scripts are removed from the transcript.

## Home page

The home page lists the catalogue 24 products at a time, with category filters (`GET /api/categories`, `GET /api/products?category=&offset=&limit=`). Searching hides the filters; clearing the search box shows them again.

## Known limitations

- YOLOv8n (COCO classes) rarely detects catalogue items on plain backgrounds; the customer can drag a box or use the whole image.
- Some Vietnamese queries miss when product names are brand-only, e.g. "nước hoa" (perfume products are named "Chanel …", "Dior …").
- A photo containing several products returns "No matching products" unless one product is boxed.
