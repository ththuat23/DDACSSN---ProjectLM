# Báo cáo tiến độ tuần 3 - Dự án Language Model Decoding

## 1. Mục tiêu

Dự án thử nghiệm và kiểm tra các chiến lược giải mã văn bản trên mô hình GPT-2:

- Greedy Search
- Beam Search
- Temperature Sampling
- Top-k Sampling
- Top-p (Nucleus) Sampling

Các thuật toán lõi được cài đặt trực tiếp trên logits và phân phối xác suất lấy từ
`model(input_ids).logits`. Chương trình không sử dụng `model.generate()` để thay thế
các bước giải mã.

## 2. Nội dung đã hoàn thành

- Tải GPT-2 và tokenizer bằng `models/model_loader.py`.
- Cài đặt các hàm giải mã trong thư mục `decoding/`.
- Tách hàm xử lý logits để kiểm thử riêng không cần sinh văn bản.
- Tạo bộ 80 prompt tiếng Anh trong `data/prompts.json`, chia thành 4 chủ đề:
  kể chuyện, kiến thức phổ thông, lập luận/gợi ý và sáng tạo/hội thoại.
- Viết unit test logits trong thư mục `tests/`:
  kiểm tra Greedy, Temperature, Top-k, Top-p và Beam Search.
- Chạy smoke test 5 prompt đầu tiên với 5 phương pháp, lưu 25 kết quả tại
  `results/smoke_test.json`.

Các kết quả này chủ yếu dùng để xác nhận quy trình và phát hiện lỗi, chưa phải kết quả
đánh giá đầy đủ trên toàn bộ 80 prompt.

## 3. Yêu cầu và cài đặt

Yêu cầu: Python 3.10 trở lên, PyTorch và Hugging Face Transformers.

```powershell
cd d:\Project-LM\decoding-project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch transformers streamlit jupyter
```

Kiểm tra môi trường:

```powershell
python check_environment.py
```

## 4. Cách chạy chương trình

Chạy ứng dụng giao diện Streamlit:

```powershell
python -m streamlit run app.py
```

Mở địa chỉ hiển thị trong terminal, mặc định là `http://localhost:8501`.

Chạy thử mô hình và các phương pháp giải mã:

```powershell
python test_model.py
python test_greedy.py
python test_decoding_all.py
python test_beam.py
python test_top_k.py
python test_top_p.py
```

Chạy smoke test và lưu kết quả:

```powershell
python run_experiments.py
```

Kiểm tra phần xử lý logits không cần tải GPT-2:

```powershell
python -m unittest discover -s tests -p "test*.py"
```

Chạy notebook:

```powershell
python -m jupyter notebook experiments\experiment.ipynb
```

## 5. Kết quả tuần 3

- Môi trường, GPT-2 và tokenizer đã chạy được.
- Các hàm giải mã và unit test logits đã được tạo.
- Bộ 80 prompt và quy trình smoke test đã sẵn sàng.
- Latency đo ban đầu cho mỗi lần sinh 50 token mới: Greedy khoảng 7.6 giây,
  Beam Search khoảng 36.5 giây, Temperature/Top-k khoảng 7.9 giây và Top-p
  khoảng 8.5 giây. Đây chỉ là kết quả thử riêng, phụ thuộc cấu hình máy.

## 6. Công việc tiếp theo

- Hoàn thiện toàn bộ so sánh trên 80 prompt.
- Cố định seed và thống nhất độ dài sinh, hardware, warm-up và EOS.
- Đo latency, throughput và peak memory trên cùng thiết lập.
- Hoàn thiện Distinct-n, Repetition Rate và Perplexity.
- Xử lý phần lặp và kiểm tra đầu ra chỉ chứa phần văn bản mới sinh.
