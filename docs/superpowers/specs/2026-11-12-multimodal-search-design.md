# Multimodal Product Search — Implementation Design

Ngày: 2026-11-12 · Nguồn thiết kế: báo cáo "E-Commerce Multimodal Product Search — System Analysis and Design Report" (Figure 4.3 design class diagram, Figure 4.4–4.6 sequence diagrams, Figure 4.7 component diagram).

## 1. Mục tiêu

Một prototype chạy trên laptop, hiện thực ba use case Search by Keyword (UC03.1), Search by Voice (UC03.2) và Search by Image (UC03.3) đúng theo thiết kế, đủ để demo và chụp màn hình cho chương Implementation của báo cáo.

**Thành công khi:**
- Một lệnh cài đặt, một lệnh seed dữ liệu, một lệnh chạy server; mở `http://localhost:8000` là dùng được.
- "laptop" và "giày chạy bộ" đều ra sản phẩm phù hợp; nói "điện thoại" ra điện thoại sau khi xác nhận transcript; upload ảnh một đôi giày ra giày.
- Mọi nhánh lỗi trong sequence diagram hiển thị đúng thông báo.
- `pytest` chạy xanh mà không cần tải model.

**Ngoài phạm vi:** đăng nhập/đăng ký, giỏ hàng, đặt hàng, trang quản trị, deploy. Bảng CARTS, CART_ITEMS, ORDERS, ORDER_ITEMS vẫn được tạo cho khớp data model nhưng không có chức năng.

## 2. Quyết định công nghệ

| Hạng mục | Lựa chọn | Ghi chú |
| --- | --- | --- |
| Backend | Python 3.14, FastAPI, Uvicorn | |
| Frontend | HTML + JavaScript thuần, phục vụ bởi FastAPI | Không build step |
| Database | SQLite (`data/shop.db`) qua SQLAlchemy 2 | **Lệch thiết kế:** thay PostgreSQL + pgvector; ghi trong báo cáo |
| Vector index | numpy, lưu `data/vectors.npz`, sau class `VectorIndex` | Đổi sang pgvector chỉ cần viết lại class này |
| Speech-to-text | `openai-whisper`, model `small` | Tự nhận diện ngôn ngữ (vi/en) |
| Object detection | `ultralytics` YOLOv8n | Đề xuất khung vật thể |
| Image embedding | CLIP `clip-ViT-B-32` (sentence-transformers) | 512 chiều |
| Text embedding | `clip-ViT-B-32-multilingual-v1` (sentence-transformers) | Cùng không gian vector với ảnh CLIP; hỗ trợ tiếng Việt |
| Dữ liệu demo | DummyJSON `https://dummyjson.com/products?limit=0` | ~190 sản phẩm, ảnh thumbnail tải về `data/images/` |
| Test | pytest + FastAPI TestClient, encoder giả | |

Model chạy trên CUDA nếu có, ngược lại CPU; nạp một lần khi khởi động (lazy, có khóa). Customer demo cố định `customer_id = 1` thay cho đăng nhập.

## 3. Cấu trúc mã nguồn

```
Project/
├── app/
│   ├── main.py                       # tạo FastAPI app, mount static, include router
│   ├── config.py                     # đường dẫn, ngưỡng, giới hạn
│   ├── presentation/static/
│   │   ├── index.html, styles.css
│   │   ├── search_ui.js              # SearchUI
│   │   ├── voice_search_ui.js        # VoiceSearchUI
│   │   ├── image_search_ui.js        # ImageSearchUI
│   │   └── permission_manager.js     # PermissionManager
│   ├── control/
│   │   ├── search_controller.py      # SearchController (APIRouter)
│   │   ├── search_service.py         # SearchService
│   │   ├── speech_to_text_service.py # SpeechToTextService
│   │   ├── image_analysis_service.py # ImageAnalysisService
│   │   └── embedding.py              # Embedder protocol + CLIP implementation
│   ├── data_access/
│   │   ├── database.py               # engine, session
│   │   ├── product_repository.py     # ProductRepository
│   │   ├── search_history_repository.py # SearchHistoryRepository
│   │   └── vector_index.py           # VectorIndex
│   └── domain/models.py              # 10 entity theo Bảng 4.2
├── scripts/seed.py
├── tests/
├── data/                             # tạo khi seed; không đưa vào git
├── requirements.txt
└── README.md
```

Mỗi class trong Figure 4.3 có cùng tên; method public giữ tên trong diagram (Python dùng snake_case, ví dụ `searchByKeyword` → `search_by_keyword`, ghi chú tên gốc trong docstring).

## 4. Dữ liệu

- `domain/models.py`: CUSTOMERS, CATEGORIES, PRODUCTS, CARTS, CART_ITEMS, ORDERS, ORDER_ITEMS, SEARCH_HISTORIES, VOICE_SEARCH_QUERIES, IMAGE_SEARCH_QUERIES với cột và khóa như Bảng 4.2 (tiền = Numeric(10,2), thời gian = DateTime). `search_type` ∈ {KEYWORD, VOICE, IMAGE}; `search_id` unique ở hai bảng chi tiết.
- `scripts/seed.py`: tạo bảng, tạo customer demo, tải sản phẩm DummyJSON (name ← title, description, price, image_url ← đường dẫn ảnh local, stock, category), tải thumbnail, tính cho mỗi sản phẩm vector ảnh và vector text (`title + category + description`), lưu `vectors.npz` gồm `product_ids`, `image_vecs`, `text_vecs` (đã chuẩn hóa L2). Chạy lại được (xóa và tạo lại).
- File upload (ảnh, audio) lưu ở `data/uploads/` với tên UUID; đường dẫn ghi vào bảng chi tiết khi search thành công.

## 5. API (SearchController)

| Endpoint | Method thiết kế | Vào | Ra |
| --- | --- | --- | --- |
| `GET /api/suggestions` | getRecentSearches, getTrendingKeywords | – | `{recent: [..5], trending: [..5]}` |
| `GET /api/search/keyword?q=` | searchByKeyword | q | `{products: [...]}` |
| `POST /api/voice/transcribe` | convertSpeechToText | multipart `audio` (webm/wav) | `{audio_id, text, language}` |
| `POST /api/search/voice` | submitQuery | JSON `{text, audio_id}` | `{products: [...]}` |
| `POST /api/image/analyze` | extractFeatures (preprocess, detectObjects) | multipart `image` | `{image_id, width, height, objects: [{label, confidence, box:[x1,y1,x2,y2]}]}` |
| `POST /api/search/image` | searchSimilar | JSON `{image_id, box?}` | `{products: [...]}` |
| `GET /api/products/{id}` | (View Product) | – | chi tiết sản phẩm |

Product trả về: `id, name, description, price, image_url, stock, category, score`.

## 6. Luồng và xếp hạng

**Keyword (Figure 4.4).** Validate (sau khi trim, 2–50 ký tự) → `normalizeKeyword` (chữ thường, bỏ dấu tiếng Việt, gộp khoảng trắng) → tập A = sản phẩm có name/description/category (đã normalize) chứa keyword, xếp theo vị trí khớp (tên trước mô tả); tập B = top-K theo cosine(vector text truy vấn, max(image_vec, text_vec) của sản phẩm) vượt ngưỡng `SEMANTIC_MIN` → kết quả = A rồi B (bỏ trùng), tối đa 24. Có kết quả → lưu SearchHistory(KEYWORD, query_text).

**Voice (Figure 4.5).** Transcribe: lưu audio, Whisper `transcribe` (fp16 nếu GPU) → text rỗng hoặc `no_speech_prob > 0.6` → 422 "Please speak again". Trả transcript; frontend hiển thị ô sửa được với nút Confirm / Speak again. Chỉ khi Confirm mới gọi `/api/search/voice` → cùng thuật toán keyword → lưu SearchHistory(VOICE) + VoiceSearchQuery(audio_path, recognized_text).

**Image (Figure 4.6).** Analyze: kiểm tra content-type và magic bytes JPG/PNG, ≤ 10 MB → `preprocessImage` (EXIF transpose, RGB, giới hạn cạnh dài 1024) → `detectObjects` (YOLOv8n, confidence ≥ 0.35, tối đa 8) → trả khung. Frontend vẽ khung trên preview; khách bấm chọn khung hoặc kéo khung riêng, hoặc dùng cả ảnh; Confirm → `/api/search/image` → cắt vùng → CLIP image vector → cosine với image_vecs → giữ ≥ `IMAGE_MIN`, top 12 → lưu SearchHistory(IMAGE, image_url) + ImageSearchQuery(image_path, extracted_features = JSON vector).

Ngưỡng `SEMANTIC_MIN`, `IMAGE_MIN` đặt trong `config.py`, chỉnh sau smoke test.

## 7. Xử lý lỗi

| Tình huống | HTTP | Thông báo hiển thị |
| --- | --- | --- |
| Keyword sai độ dài | 400 | Keyword must be 2-50 characters |
| Ảnh sai định dạng/quá lớn | 400 | Only JPG/PNG, max 10MB |
| image_id/audio_id không tồn tại | 404 | Upload expired, please try again |
| Không nghe ra chữ | 422 | Please speak again |
| Không có sản phẩm (keyword/voice) | 200, rỗng | No products found |
| Không có sản phẩm (image) | 200, rỗng | No matching products |
| Model không nạp được | 503 | Voice/Image search is unavailable, please type a keyword |
| Bị từ chối quyền mic/camera | – (frontend) | Permission denied. Please type a keyword instead. |

Frontend validate trước (keyword, file) nhưng backend luôn kiểm tra lại. Keyword search không phụ thuộc model: nếu encoder text lỗi, chỉ trả tập A.

## 8. Kiểm thử

- **Unit:** normalize (bỏ dấu, "Giày  Chạy" → "giay chay"), validate keyword (1, 2, 50, 51 ký tự, toàn khoảng trắng), validate ảnh (png/jpg hợp lệ, gif, file đổi đuôi, > 10 MB), VectorIndex (thứ tự, ngưỡng, k), SearchService lưu đúng loại lịch sử và không lưu khi rỗng, gợi ý recent/trending.
- **Integration API:** TestClient + SQLite tạm + `FakeEmbedder`/`FakeSpeech`/`FakeDetector` (inject qua dependency); mỗi endpoint một ca thành công và các ca lỗi ở mục 7.
- **Smoke thủ công:** sau seed, chạy server với model thật, thử 4 truy vấn ở mục 1, chụp màn hình.

## 9. Rủi ro

- Lần đầu tải ~1–2 GB (torch CUDA, Whisper small ~460 MB, CLIP ~600 MB, YOLOv8n ~6 MB).
- Gói nào chưa có wheel cho Python 3.14 → dùng phiên bản thay thế hoặc cài Python 3.12 trong venv; quyết định khi cài.
- Ghi âm trên trình duyệt chỉ hoạt động ở `localhost` hoặc HTTPS.
