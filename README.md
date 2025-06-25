# 🎓 Machine Learning Graduation Project

## Content-Based Company Similarity Recommendation and 'Recommend or Not' Classification for Candidates

### 📋 Mô tả dự án

Dự án này bao gồm hai mô hình machine learning chính:

1. **Content-Based Company Similarity**: Gợi ý các công ty tương tự dựa trên đặc điểm và nội dung
2. **Candidate Classification**: Phân loại ứng viên có nên được gợi ý hay không

### 🛠️ Cài đặt và Chạy

#### 1. Clone repository và setup

```bash
git clone <your-repo-url>
cd ml_graduation_project
```

#### 2. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

#### 4. Tạo dữ liệu mẫu và models

```bash
# Tạo dữ liệu CSV mẫu
python create_sample_data.py

# Tạo models mẫu (nếu chưa có models thật)
python create_sample_models.py
```

#### 5. Chạy ứng dụng

```bash
streamlit run app.py
```

Truy cập: `http://localhost:8501`

### 📁 Cấu trúc thư mục

```
ml_graduation_project/
│
├── app.py                          # File chính Streamlit
├── requirements.txt                # Dependencies
├── README.md                      # File này
├── create_sample_data.py          # Script tạo dữ liệu mẫu
├── create_sample_models.py        # Script tạo models mẫu
│
├── models/
│   ├── company_similarity_model.pkl
│   └── candidate_recommendation_model.pkl
│
├── data/
│   ├── companies_data.csv          # Dữ liệu 100 công ty
│   ├── candidates_data.csv         # Dữ liệu 500 ứng viên
│   ├── sample_predictions.csv      # 200 predictions mẫu
│   └── job_positions.csv          # 50 vị trí việc làm
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # Load dữ liệu
│   ├── model_loader.py            # Load models
│   └── preprocessing.py           # Xử lý dữ liệu
│
└── assets/
    └── images/
```

### 🚀 Tính năng chính

#### 🏢 Company Similarity Recommendation
- Tìm kiếm công ty tương tự dựa trên đặc điểm
- Hiển thị điểm similarity và thông tin chi tiết
- Biểu đồ trực quan hóa kết quả

#### 👤 Candidate Classification
- Phân loại ứng viên: Recommend/Not Recommend
- Hiển thị confidence score và feature importance
- Phân tích chi tiết các yếu tố ảnh hướng

#### 📊 Dashboard và Analytics
- Tổng quan performance của models
- Metrics và charts
- Responsive design

### 🔧 Thay thế bằng dữ liệu thật

#### 1. Thay thế models
```python
# Lưu model thật của bạn
import pickle

# Company similarity model
with open('models/company_similarity_model.pkl', 'wb') as f:
    pickle.dump(your_similarity_model, f)

# Candidate classification model  
with open('models/candidate_recommendation_model.pkl', 'wb') as f:
    pickle.dump(your_classification_model, f)
```

#### 2. Thay thế dữ liệu
- Thay thế các file CSV trong thư mục `data/`
- Đảm bảo format columns giống như sample data
- Cập nhật `utils/preprocessing.py` nếu cần

#### 3. Tùy chỉnh preprocessing
- Sửa đổi `utils/preprocessing.py` theo yêu cầu models
- Cập nhật feature engineering logic
- Điều chỉnh input/output format

### 📊 Dữ liệu mẫu

#### Companies (100 records)
- company_id, company_name, industry, company_size
- location, founded_year, employee_count, revenue
- description, website, benefits

#### Candidates (500 records)  
- candidate_id, full_name, age, gender, experience_years
- education_level, major, skills, salary_expectation
- preferred_location, job_type_preference, english_level

#### Predictions (200 records)
- prediction_id, candidate_id, company_id
- recommendation_score, predicted_class, confidence

### 🎨 Tùy chỉnh giao diện

#### CSS Styling
- Sửa đổi CSS trong `app.py`
- Thêm custom styles trong `assets/style.css`

#### Colors và Themes
- Thay đổi color scheme trong CSS
- Cập nhật Plotly chart colors
- Điều chỉnh Streamlit theme

### 🛠️ Development

#### Thêm tính năng mới
1. Tạo file trong `pages/` cho trang mới
2. Import và thêm vào navigation trong `app.py`
3. Cập nhật `utils/` nếu cần functions mới

#### Debug và Testing
```bash
# Chạy với verbose logging
streamlit run app.py --logger.level=debug

# Test individual components
python -m pytest tests/ (nếu có tests)
```

### 📝 TODO cho production

- [ ] Thay thế fake data bằng real data
- [ ] Thay thế demo models bằng trained models  
- [ ] Thêm user authentication
- [ ] Database integration
- [ ] API endpoints
- [ ] Error handling improvements
- [ ] Performance optimization
- [ ] Unit tests
- [ ] Documentation updates

### 📧 Liên hệ

**Sinh viên**: [Tên của bạn]  
**MSSV**: [Mã số sinh viên]  
**Email**: [Email của bạn]  
**Trường**: [Tên trường]

---

## 🚀 Quick Start Commands

```bash
# Setup project
git clone <repo> && cd ml_graduation_project
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Create sample data & models
python create_sample_data.py
python create_sample_models.py

# Run application
streamlit run app.py
```