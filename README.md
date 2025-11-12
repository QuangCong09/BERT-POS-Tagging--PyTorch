# 🤖 BERT Fine-tuning for Part-of-Speech (POS) Tagging with PyTorch

Dự án này tập trung vào việc áp dụng kiến trúc Deep Learning hiện đại (Transformer/BERT) để giải quyết bài toán Phân loại Từ loại (Part-of-Speech Tagging - POS Tagging).

## 💡 Mục Tiêu Dự Án

Mục tiêu chính là tinh chỉnh một mô hình BERT đã được huấn luyện trước (Pre-trained) cho tác vụ Token Classification trên dữ liệu chuyên biệt. Dự án thể hiện kỹ năng xây dựng pipeline huấn luyện Deep Learning từ đầu đến cuối, bao gồm xử lý dữ liệu, tinh chỉnh mô hình, và đánh giá hiệu suất.

## 🚀 Công Nghệ Chính (Tech Stack)

* **Lập trình:** Python
* **Deep Learning:** **PyTorch** (Core Framework)
* **NLP/Transformer:** **Hugging Face Transformers** (sử dụng thư viện `transformers` và `AutoModelForTokenClassification`)
* **Tiền xử lý:** NLTK, Scikit-learn, Collections
* **Đánh giá:** `evaluate` library (Hugging Face)

## 🗂️ Tập Dữ Liệu (Dataset)

* **Nguồn:** Sử dụng tập dữ liệu **Treebank** từ thư viện **NLTK** (Natural Language Toolkit), bao gồm các câu đã được gán nhãn từ loại.
* **Phân chia:** Dữ liệu được chia thành ba tập:
    * Tập huấn luyện (Train): 70%
    * Tập kiểm tra (Validation): 15%
    * Tập đánh giá cuối cùng (Test): 15%

## 🛠️ Triển Khai Kỹ Thuật (Key Implementations)

1.  **Custom PyTorch Dataset:** Phát triển lớp `PosTagging_Dataset` kế thừa từ `torch.utils.data.Dataset` để:
    * Xử lý tuần tự đầu vào (`input_ids`) và nhãn (`labels`).
    * Thực hiện **Padding (Đệm)** và **Truncation (Cắt ngắn)** để chuẩn hóa độ dài chuỗi.
    * Ánh xạ nhãn từ chuỗi ký tự sang ID số.
2.  **Xử lý Nhãn Đệm:** Sử dụng nhãn **"O" (Outside)** cho các token đệm trong quá trình xử lý, đảm bảo mô hình không tính toán lỗi trên các phần đã được đệm.
3.  **Fine-tuning:** Tinh chỉnh mô hình **`QCRI/bert-base-multilingual-cased-pos-english`** trong 10 epochs.
4.  **Đánh giá Chính xác:** Thiết lập hàm `compute_metrics` tùy chỉnh để tính toán **Accuracy** chính xác, sử dụng **Masking** để loại trừ các nhãn padding khỏi kết quả đánh giá cuối cùng.


## ⚙️ Hướng Dẫn Chạy Dự Án

### 1. Cài đặt Môi trường

```bash
# Clone repository này về máy
git clone [https://github.com/your-username/BERT-POS-Tagging-PyTorch.git](https://github.com/your-username/BERT-POS-Tagging-PyTorch.git)
cd BERT-POS-Tagging-PyTorch

# Cài đặt các thư viện cần thiết
pip install torch transformers scikit-learn nltk evaluate numpy
