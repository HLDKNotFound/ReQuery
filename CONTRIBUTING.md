# Contributing to ReQuery

Cảm ơn bạn đã quan tâm đóng góp cho dự án **ReQuery**! Dưới đây là hướng dẫn quy trình phát triển và kiểm thử mã nguồn.

---

## 1. Thiết lập môi trường phát triển (Development Setup)

1. Clone repository và chuyển vào thư mục:
   ```bash
   git clone <repo-url>
   cd ReQuery
   ```

2. Kích hoạt môi trường conda hoặc tạo virtual environment:
   ```bash
   conda activate ml-env
   # Hoặc tạo virtualenv:
   # python -m venv .venv && source .venv/bin/activate
   ```

3. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

4. Tạo file cấu hình môi trường `.env`:
   ```bash
   cp .env.example .env
   ```

---

## 2. Quy trình kiểm thử (Running Tests)

Trước khi tạo pull request hoặc commit thay đổi, hãy đảm bảo toàn bộ unit test đều vượt qua:

```bash
pytest tests/
```

Bộ unit test bao gồm:
- `tests/test_preprocessing.py`: Kiểm thử chuẩn hóa văn bản, trích xuất dữ liệu và lọc dữ liệu trùng lặp.
- `tests/test_fusion.py`: Kiểm thử các thuật toán hợp nhất bảng xếp hạng (Reciprocal Rank Fusion, Max-Score, Mean-Score).
- `tests/test_metrics.py`: Kiểm thử tính toán Recall@K, MRR, nDCG, Hit Rate và các chỉ số đa dạng truy vấn.

---

## 3. Cấu trúc mã nguồn (Code Structure)

Mã nguồn được tổ chức theo cấu trúc module rõ ràng:
- `configs/`: Chứa các file YAML định hình siêu tham số mô hình, huấn luyện, retriever và đánh giá.
- `src/data/`: Script tải dữ liệu, tiền xử lý và tokenizer.
- `src/training/`: Mã nguồn huấn luyện QLoRA 4-bit và quản lý checkpoint.
- `src/inference/`: Bộ sinh truy vấn tối ưu hóa tìm kiếm (SLM generator).
- `src/retrieval/`: Vector database FAISS và thuật toán xếp hạng RRF.
- `src/evaluation/`: Benchmark so sánh retrieval, độ đa dạng và RAG downstream.
- `api/`: FastAPI web server.
- `frontend/`: Giao diện người dùng Web UI (HTML, CSS, JS).

---

## 4. Quy ước Commit & Code Style

- Sử dụng **Python 3.10+** với type annotations rõ ràng.
- Đảm bảo mã nguồn không chứa hard-coded paths. Mọi đường dẫn tương đối phải thông qua `src.utils.config.resolve_path`.
- Viết docstring ngắn gọn giải thích mục đích của hàm/lớp.
