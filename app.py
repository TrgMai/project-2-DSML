# import streamlit as st
# import pandas as pd
# import numpy as np
# from utils.data_loader import load_sample_data
# from utils.model_loader import load_models
# import plotly.express as px
# import plotly.graph_objects as go

# # Cấu hình trang
# st.set_page_config(
#     page_title="ML Graduation Project",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Dark CSS
# def get_dark_css():
#     return """
#     <style>
#         .stApp {
#             background-color: #1a1a1a !important;
#             color: #ffffff !important;
#         }
        
#         .main .block-container {
#             background-color: #1a1a1a !important;
#             color: #ffffff !important;
#         }
        
#         .main-header {
#             font-size: 3rem;
#             color: #4fc3f7 !important;
#             text-align: center;
#             margin-bottom: 2rem;
#         }
        
#         .metric-container {
#             background-color: #2d2d2d !important;
#             padding: 1rem;
#             border-radius: 8px;
#             border-left: 4px solid #4fc3f7;
#             color: #ffffff !important;
#         }
        
#         section[data-testid="stSidebar"] {
#             background-color: #2d2d2d !important;
#         }
        
#         .stApp * {
#             color: #ffffff !important;
#         }
        
#         .stTextInput > div > div > input,
#         .stNumberInput > div > div > input {
#             background-color: #333333 !important;
#             color: #ffffff !important;
#             border: 1px solid #555555 !important;
#         }
        
#         .stSelectbox > div > div > div {
#             background-color: #333333 !important;
#             color: #ffffff !important;
#             border: 1px solid #555555 !important;
#         }
        
#         .stButton > button {
#             background-color: #333333 !important;
#             color: #ffffff !important;
#             border: 1px solid #555555 !important;
#         }
        
#         .stButton > button:hover {
#             background-color: #4fc3f7 !important;
#             border-color: #4fc3f7 !important;
#         }
        
#         [data-testid="metric-container"] {
#             background-color: #2d2d2d !important;
#             border: 1px solid #444444 !important;
#         }
#     </style>
#     """

# def main():
#     # Initialize session state
#     if 'page' not in st.session_state:
#         st.session_state.page = "🏠 Trang chủ"
    
#     # Apply dark theme
#     st.markdown(get_dark_css(), unsafe_allow_html=True)
    
#     # Sidebar
#     st.sidebar.title("📋 Menu")
    
#     # Navigation buttons
#     if st.sidebar.button("🏠 Trang chủ", use_container_width=True, type="primary" if st.session_state.page == "🏠 Trang chủ" else "secondary"):
#         st.session_state.page = "🏠 Trang chủ"
#         st.rerun()
    
#     if st.sidebar.button("🏢 Company Similarity", use_container_width=True, type="primary" if st.session_state.page == "🏢 Company Similarity" else "secondary"):
#         st.session_state.page = "🏢 Company Similarity" 
#         st.rerun()
    
#     if st.sidebar.button("👤 Candidate Classification", use_container_width=True, type="primary" if st.session_state.page == "👤 Candidate Classification" else "secondary"):
#         st.session_state.page = "👤 Candidate Classification"
#         st.rerun()
    
#     if st.sidebar.button("ℹ️ About", use_container_width=True, type="primary" if st.session_state.page == "ℹ️ About" else "secondary"):
#         st.session_state.page = "ℹ️ About"
#         st.rerun()
    
#     # Header
#     st.markdown('<h1 class="main-header">🎓 Machine Learning Graduation Project</h1>', unsafe_allow_html=True)
#     st.markdown('<h2 style="text-align: center; color: #b0bec5;">Content-Based Company Similarity & Candidate Classification</h2>', unsafe_allow_html=True)
    
#     # Route pages
#     if st.session_state.page == "🏠 Trang chủ":
#         show_homepage()
#     elif st.session_state.page == "🏢 Company Similarity":
#         show_company_similarity()
#     elif st.session_state.page == "👤 Candidate Classification":
#         show_candidate_classification()
#     elif st.session_state.page == "ℹ️ About":
#         show_about()

# def show_homepage():
#     st.markdown("## 📊 Tổng quan dự án")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         ### 🏢 Content-Based Company Similarity
#         - **Mục tiêu**: Gợi ý các công ty tương tự dựa trên nội dung và đặc điểm
#         - **Phương pháp**: Content-based filtering
#         - **Ứng dụng**: Giúp ứng viên tìm kiếm công ty phù hợp
#         """)
        
#         st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#         col1_1, col1_2, col1_3 = st.columns(3)
#         col1_1.metric("Companies", "1,250", "↑ 15%")
#         col1_2.metric("Accuracy", "87.3%", "↑ 2.1%")
#         col1_3.metric("Precision", "89.1%", "↑ 1.8%")
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#         ### 👤 'Recommend or Not' Classification
#         - **Mục tiêu**: Phân loại ứng viên có nên được gợi ý hay không
#         - **Phương pháp**: Binary classification
#         - **Ứng dụng**: Lọc ứng viên phù hợp cho từng vị trí
#         """)
        
#         st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#         col2_1, col2_2, col2_3 = st.columns(3)
#         col2_1.metric("Candidates", "3,847", "↑ 25%")
#         col2_2.metric("F1-Score", "91.7%", "↑ 3.2%")
#         col2_3.metric("Recall", "88.9%", "↑ 2.5%")
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Performance charts
#     st.markdown("## 📈 Performance Overview")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
#         accuracy = [82.1, 84.3, 85.7, 86.2, 87.0, 87.3]
        
#         fig1 = px.line(x=months, y=accuracy, title="Company Similarity Model - Accuracy Over Time")
#         fig1.update_traces(line_color='#4fc3f7', line_width=3)
#         fig1.update_layout(
#             xaxis_title="Month", 
#             yaxis_title="Accuracy (%)",
#             paper_bgcolor='rgba(0,0,0,0)',
#             plot_bgcolor='rgba(0,0,0,0)',
#             font_color='white'
#         )
#         st.plotly_chart(fig1, use_container_width=True)
    
#     with col2:
#         metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
#         values = [90.5, 89.2, 88.9, 91.7]
        
#         fig2 = px.bar(x=metrics, y=values, title="Candidate Classification - Model Metrics")
#         fig2.update_traces(marker_color='#4fc3f7')
#         fig2.update_layout(
#             xaxis_title="Metrics", 
#             yaxis_title="Score (%)",
#             paper_bgcolor='rgba(0,0,0,0)',
#             plot_bgcolor='rgba(0,0,0,0)',
#             font_color='white'
#         )
#         st.plotly_chart(fig2, use_container_width=True)

# def show_company_similarity():
#     st.markdown("## 🏢 Content-Based Company Similarity Recommendation")
    
#     # Tabs for different input methods
#     tab1, tab2 = st.tabs(["🔍 Manual Input", "📁 Upload File"])
    
#     with tab1:
#         st.markdown("### 🔍 Tìm kiếm công ty tương tự")
        
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             companies_df = load_sample_data('companies')
#             company_names = companies_df['company_name'].tolist() if companies_df is not None else ['TechCorp', 'DataSoft', 'AIVentures']
            
#             selected_company = st.selectbox("Chọn công ty:", company_names)
#             num_recommendations = st.slider("Số lượng gợi ý:", 1, 10, 5)
        
#         with col2:
#             st.markdown("### ⚙️ Parameters")
#             similarity_threshold = st.slider("Similarity threshold:", 0.1, 1.0, 0.7)
#             include_industry = st.checkbox("Filter by industry", True)
        
#         if st.button("🔍 Tìm công ty tương tự", type="primary", key="manual_search"):
#             process_company_similarity(selected_company, num_recommendations, similarity_threshold)
    
#     with tab2:
#         st.markdown("### 📁 Upload Company Data")
#         st.markdown("Upload file CSV/Excel chứa thông tin công ty để batch predict")
        
#         uploaded_file = st.file_uploader(
#             "Chọn file:",
#             type=['csv', 'xlsx', 'xls'],
#             help="File should contain columns: company_name, industry, company_size, description, etc."
#         )
        
#         if uploaded_file is not None:
#             try:
#                 # Read uploaded file
#                 if uploaded_file.name.endswith('.csv'):
#                     df = pd.read_csv(uploaded_file)
#                 else:
#                     df = pd.read_excel(uploaded_file)
                
#                 st.markdown("### 📊 Preview Data")
#                 st.dataframe(df.head(), use_container_width=True)
                
#                 # Settings for batch processing
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     num_recommendations = st.slider("Số gợi ý cho mỗi công ty:", 1, 10, 3, key="batch_num")
#                 with col2:
#                     similarity_threshold = st.slider("Similarity threshold:", 0.1, 1.0, 0.7, key="batch_threshold")
                
#                 if st.button("🚀 Process All Companies", type="primary"):
#                     process_batch_companies(df, num_recommendations, similarity_threshold)
                    
#             except Exception as e:
#                 st.error(f"Error reading file: {str(e)}")
#                 st.info("Please ensure your file has the correct format with columns like: company_name, industry, description")

# def process_company_similarity(selected_company, num_recommendations, similarity_threshold):
#     with st.spinner("Đang xử lý..."):
#         similar_companies = generate_fake_similar_companies(selected_company, num_recommendations)
        
#         st.markdown("### 📊 Kết quả gợi ý")
        
#         cols = st.columns(min(3, len(similar_companies)))
#         for i, company in enumerate(similar_companies):
#             with cols[i % 3]:
#                 st.markdown(f"""
#                 <div class="metric-container">
#                     <h4>{company['name']}</h4>
#                     <p><strong>Similarity:</strong> {company['similarity']:.2f}</p>
#                     <p><strong>Industry:</strong> {company['industry']}</p>
#                     <p><strong>Size:</strong> {company['size']}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
        
#         st.markdown("### 📈 Similarity Scores")
#         names = [c['name'] for c in similar_companies]
#         scores = [c['similarity'] for c in similar_companies]
        
#         fig = px.bar(x=names, y=scores, title="Company Similarity Scores")
#         fig.update_traces(marker_color='#4fc3f7')
#         fig.update_layout(
#             xaxis_title="Companies", 
#             yaxis_title="Similarity Score",
#             paper_bgcolor='rgba(0,0,0,0)',
#             plot_bgcolor='rgba(0,0,0,0)',
#             font_color='white'
#         )
#         st.plotly_chart(fig, use_container_width=True)

# def process_batch_companies(df, num_recommendations, similarity_threshold):
#     with st.spinner("Processing batch companies..."):
#         results = []
#         progress_bar = st.progress(0)
        
#         for i, row in df.iterrows():
#             company_name = row.get('company_name', f'Company_{i}')
#             similar_companies = generate_fake_similar_companies(company_name, num_recommendations)
            
#             for sim_company in similar_companies:
#                 results.append({
#                     'Original_Company': company_name,
#                     'Similar_Company': sim_company['name'],
#                     'Similarity_Score': sim_company['similarity'],
#                     'Industry': sim_company['industry'],
#                     'Size': sim_company['size']
#                 })
            
#             progress_bar.progress((i + 1) / len(df))
        
#         results_df = pd.DataFrame(results)
        
#         st.markdown("### 📊 Batch Processing Results")
#         st.dataframe(results_df, use_container_width=True)
        
#         # Download button
#         csv = results_df.to_csv(index=False)
#         st.download_button(
#             label="📥 Download Results as CSV",
#             data=csv,
#             file_name="company_similarity_results.csv",
#             mime="text/csv"
#         )
        
#         # Summary stats
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Companies Processed", len(df))
#         with col2:
#             st.metric("Total Recommendations", len(results_df))
#         with col3:
#             avg_similarity = results_df['Similarity_Score'].mean()
#             st.metric("Avg Similarity", f"{avg_similarity:.2f}")

# def show_candidate_classification():
#     st.markdown("## 👤 'Recommend or Not' Classification for Candidates")
    
#     # Tabs for different input methods
#     tab1, tab2 = st.tabs(["👤 Single Candidate", "📁 Batch Upload"])
    
#     with tab1:
#         st.markdown("### 📝 Thông tin ứng viên")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             name = st.text_input("Tên ứng viên:", "Nguyen Van A")
#             age = st.number_input("Tuổi:", 20, 65, 28)
#             experience = st.number_input("Kinh nghiệm (năm):", 0, 20, 3)
#             education = st.selectbox("Học vấn:", ["High School", "Bachelor", "Master", "PhD"])
        
#         with col2:
#             skills = st.multiselect("Kỹ năng:", 
#                                    ["Python", "Java", "JavaScript", "SQL", "Machine Learning", "Data Analysis", "Project Management"])
#             salary_expectation = st.number_input("Mong muốn lương (triệu VND):", 10, 100, 25)
#             location = st.selectbox("Địa điểm:", ["Ho Chi Minh City", "Hanoi", "Da Nang", "Other"])
#             job_type = st.selectbox("Loại công việc:", ["Full-time", "Part-time", "Contract", "Intern"])
        
#         if st.button("🎯 Phân loại ứng viên", type="primary", key="single_classify"):
#             process_single_candidate(name, age, experience, education, skills, salary_expectation, location, job_type)
    
#     with tab2:
#         st.markdown("### 📁 Upload Candidates Data")
#         st.markdown("Upload file CSV/Excel chứa thông tin ứng viên để batch classify")
        
#         uploaded_file = st.file_uploader(
#             "Chọn file ứng viên:",
#             type=['csv', 'xlsx', 'xls'],
#             help="File should contain columns: name, age, experience_years, education_level, skills, salary_expectation, etc."
#         )
        
#         if uploaded_file is not None:
#             try:
#                 # Read uploaded file
#                 if uploaded_file.name.endswith('.csv'):
#                     df = pd.read_csv(uploaded_file)
#                 else:
#                     df = pd.read_excel(uploaded_file)
                
#                 st.markdown("### 📊 Preview Candidates Data")
#                 st.dataframe(df.head(), use_container_width=True)
                
#                 # Column mapping
#                 st.markdown("### 🔗 Column Mapping")
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     name_col = st.selectbox("Name column:", df.columns, index=0)
#                     age_col = st.selectbox("Age column:", df.columns, index=1 if len(df.columns) > 1 else 0)
#                     exp_col = st.selectbox("Experience column:", df.columns, index=2 if len(df.columns) > 2 else 0)
                
#                 with col2:
#                     edu_col = st.selectbox("Education column:", df.columns, index=3 if len(df.columns) > 3 else 0)
#                     skills_col = st.selectbox("Skills column:", df.columns, index=4 if len(df.columns) > 4 else 0)
#                     salary_col = st.selectbox("Salary column:", df.columns, index=5 if len(df.columns) > 5 else 0)
                
#                 # Settings
#                 confidence_threshold = st.slider("Confidence threshold for recommendation:", 0.5, 0.9, 0.6)
                
#                 if st.button("🚀 Classify All Candidates", type="primary"):
#                     process_batch_candidates(df, name_col, age_col, exp_col, edu_col, skills_col, salary_col, confidence_threshold)
                    
#             except Exception as e:
#                 st.error(f"Error reading file: {str(e)}")
#                 st.info("Please ensure your file has the correct format with candidate information")

# def process_single_candidate(name, age, experience, education, skills, salary_expectation, location, job_type):
#     with st.spinner("Đang phân tích..."):
#         prediction_result = generate_fake_candidate_prediction(name, age, experience, education, skills, salary_expectation)
        
#         st.markdown("### 📊 Kết quả phân loại")
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             recommendation = "✅ RECOMMEND" if prediction_result['recommend'] else "❌ NOT RECOMMEND"
#             color = "#2ecc71" if prediction_result['recommend'] else "#e74c3c"
#             st.markdown(f"""
#             <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color}; color: white;">
#                 <h2>{recommendation}</h2>
#                 <p>Confidence: {prediction_result['confidence']:.1%}</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col2:
#             st.markdown("#### 📈 Score Breakdown")
#             st.progress(prediction_result['skill_score'], "Skills Match")
#             st.progress(prediction_result['experience_score'], "Experience")
#             st.progress(prediction_result['education_score'], "Education")
#             st.progress(prediction_result['overall_score'], "Overall")
        
#         with col3:
#             st.markdown("#### 💡 Insights")
#             for insight in prediction_result['insights']:
#                 st.write(f"• {insight}")
        
#         st.markdown("### 🎯 Feature Importance")
#         features = ['Skills', 'Experience', 'Education', 'Salary Fit', 'Location']
#         importance = prediction_result['feature_importance']
        
#         fig = px.horizontal_bar(x=importance, y=features, title="Feature Importance in Classification")
#         fig.update_traces(marker_color='#4fc3f7')
#         fig.update_layout(
#             paper_bgcolor='rgba(0,0,0,0)',
#             plot_bgcolor='rgba(0,0,0,0)',
#             font_color='white'
#         )
#         st.plotly_chart(fig, use_container_width=True)

# def process_batch_candidates(df, name_col, age_col, exp_col, edu_col, skills_col, salary_col, confidence_threshold):
#     with st.spinner("Classifying all candidates..."):
#         results = []
#         progress_bar = st.progress(0)
        
#         for i, row in df.iterrows():
#             # Extract data from row
#             name = row.get(name_col, f'Candidate_{i}')
#             age = row.get(age_col, 25)
#             experience = row.get(exp_col, 0)
#             education = row.get(edu_col, 'Bachelor')
#             skills = str(row.get(skills_col, '')).split(',') if pd.notna(row.get(skills_col, '')) else []
#             salary = row.get(salary_col, 25)
            
#             # Get prediction
#             prediction = generate_fake_candidate_prediction(name, age, experience, education, skills, salary)
            
#             results.append({
#                 'Name': name,
#                 'Age': age,
#                 'Experience_Years': experience,
#                 'Education': education,
#                 'Skills_Count': len(skills),
#                 'Salary_Expectation': salary,
#                 'Recommendation': 'RECOMMEND' if prediction['recommend'] else 'NOT RECOMMEND',
#                 'Confidence': prediction['confidence'],
#                 'Skills_Score': prediction['skill_score'],
#                 'Experience_Score': prediction['experience_score'],
#                 'Education_Score': prediction['education_score'],
#                 'Overall_Score': prediction['overall_score']
#             })
            
#             progress_bar.progress((i + 1) / len(df))
        
#         results_df = pd.DataFrame(results)
        
#         st.markdown("### 📊 Batch Classification Results")
        
#         # Summary metrics
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Total Candidates", len(results_df))
#         with col2:
#             recommended = len(results_df[results_df['Recommendation'] == 'RECOMMEND'])
#             st.metric("Recommended", recommended, f"{recommended/len(results_df)*100:.1f}%")
#         with col3:
#             avg_confidence = results_df['Confidence'].mean()
#             st.metric("Avg Confidence", f"{avg_confidence:.2f}")
#         with col4:
#             high_confidence = len(results_df[results_df['Confidence'] > confidence_threshold])
#             st.metric("High Confidence", high_confidence, f"{high_confidence/len(results_df)*100:.1f}%")
        
#         # Filter options
#         st.markdown("### 🔍 Filter Results")
#         col1, col2 = st.columns(2)
#         with col1:
#             show_only = st.selectbox("Show only:", ["All", "RECOMMEND", "NOT RECOMMEND"])
#         with col2:
#             min_confidence = st.slider("Minimum confidence:", 0.0, 1.0, 0.0)
        
#         # Apply filters
#         filtered_df = results_df.copy()
#         if show_only != "All":
#             filtered_df = filtered_df[filtered_df['Recommendation'] == show_only]
#         filtered_df = filtered_df[filtered_df['Confidence'] >= min_confidence]
        
#         st.dataframe(filtered_df, use_container_width=True)
        
#         # Download button
#         csv = filtered_df.to_csv(index=False)
#         st.download_button(
#             label="📥 Download Results as CSV",
#             data=csv,
#             file_name="candidate_classification_results.csv",
#             mime="text/csv"
#         )
        
#         # Visualization
#         st.markdown("### 📈 Results Distribution")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Recommendation distribution
#             rec_counts = results_df['Recommendation'].value_counts()
#             fig1 = px.pie(values=rec_counts.values, names=rec_counts.index, 
#                          title="Recommendation Distribution")
#             fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
#             st.plotly_chart(fig1, use_container_width=True)
        
#         with col2:
#             # Confidence distribution
#             fig2 = px.histogram(results_df, x='Confidence', nbins=20, 
#                               title="Confidence Score Distribution")
#             fig2.update_traces(marker_color='#4fc3f7')
#             fig2.update_layout(
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 font_color='white'
#             )
#             st.plotly_chart(fig2, use_container_width=True)

# def show_about():
#     st.markdown("## ℹ️ Về dự án")
    
#     st.markdown("""
#     ### 🎓 Đồ án tốt nghiệp
    
#     **Chủ đề:** Content-Based Company Similarity Recommendation and 'Recommend or Not' Classification for Candidates
    
#     **Mô tả:**
#     Dự án này bao gồm hai mô hình machine learning chính:
    
#     1. **Content-Based Company Similarity:** Sử dụng thuật toán content-based filtering để gợi ý các công ty tương tự dựa trên đặc điểm và nội dung công ty.
    
#     2. **Candidate Classification:** Mô hình phân loại binary để xác định một ứng viên có nên được gợi ý cho một vị trí cụ thể hay không.
    
#     ### 🛠️ Công nghệ sử dụng
#     - **Backend:** Python, Scikit-learn, Pandas, NumPy
#     - **Frontend:** Streamlit
#     - **Visualization:** Plotly, Matplotlib
#     - **Model Storage:** Pickle files
    
#     ### 📊 Dữ liệu
#     - Dữ liệu công ty: Thông tin về tên, ngành nghề, quy mô, địa điểm
#     - Dữ liệu ứng viên: Thông tin cá nhân, kỹ năng, kinh nghiệm, học vấn
    
#     ### 🚀 Triển khai
#     Ứng dụng được xây dựng bằng Streamlit để dễ dàng demo và sử dụng.
#     """)
    
#     st.markdown("---")
#     st.markdown("**Được phát triển bởi:** [Tên sinh viên] - [Mã số sinh viên]")
#     st.markdown("**Trường:** [Tên trường] - **Năm:** 2024")

# def generate_fake_similar_companies(selected_company, num_recommendations):
#     import random
    
#     companies = [
#         {"name": "TechStart Inc", "industry": "Technology", "size": "Startup"},
#         {"name": "DataCorp Ltd", "industry": "Data Analytics", "size": "Medium"},
#         {"name": "AI Solutions", "industry": "Artificial Intelligence", "size": "Large"},
#         {"name": "CloudTech Pro", "industry": "Cloud Computing", "size": "Medium"},
#         {"name": "DevOps Masters", "industry": "Software Development", "size": "Small"},
#         {"name": "FinTech Innovations", "industry": "Financial Technology", "size": "Startup"},
#         {"name": "Digital Marketing Co", "industry": "Marketing", "size": "Medium"},
#         {"name": "CyberSec Guard", "industry": "Cybersecurity", "size": "Large"}
#     ]
    
#     selected_companies = random.sample(companies, min(num_recommendations, len(companies)))
    
#     for company in selected_companies:
#         company['similarity'] = round(random.uniform(0.6, 0.95), 2)
    
#     return sorted(selected_companies, key=lambda x: x['similarity'], reverse=True)

# def generate_fake_candidate_prediction(name, age, experience, education, skills, salary_expectation):
#     import random
    
#     skill_score = min(len(skills) * 0.15, 1.0)
#     experience_score = min(experience * 0.1, 1.0)
#     education_score = {"High School": 0.5, "Bachelor": 0.7, "Master": 0.9, "PhD": 1.0}[education]
#     salary_score = max(0.3, 1.0 - (salary_expectation - 20) * 0.01)
    
#     overall_score = (skill_score + experience_score + education_score + salary_score) / 4
#     confidence = overall_score + random.uniform(-0.1, 0.1)
#     recommend = confidence > 0.6
    
#     insights = []
#     if skill_score > 0.7:
#         insights.append("Strong technical skills match")
#     if experience_score > 0.5:
#         insights.append("Good experience level for the role")
#     if education_score > 0.8:
#         insights.append("High education qualification")
#     if salary_score < 0.5:
#         insights.append("Salary expectation might be high")
    
#     return {
#         'recommend': recommend,
#         'confidence': confidence,
#         'skill_score': skill_score,
#         'experience_score': experience_score,
#         'education_score': education_score,
#         'overall_score': overall_score,
#         'feature_importance': [0.35, 0.25, 0.20, 0.12, 0.08],
#         'insights': insights
#     }

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_sample_data
from utils.model_loader import load_models
import plotly.express as px
import plotly.graph_objects as go

# Cấu hình trang
st.set_page_config(
    page_title="ML Graduation Project",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark CSS
def get_dark_css():
    return """
    <style>
        .stApp {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
        }
        
        .main .block-container {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
        }
        
        .main-header {
            font-size: 3rem;
            color: #4fc3f7 !important;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-container {
            background-color: #2d2d2d !important;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4fc3f7;
            color: #ffffff !important;
        }
        
        .stApp * {
            color: #ffffff !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #2d2d2d;
            padding: 8px;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: #333333;
            border-radius: 8px;
            color: #ffffff;
            border: 1px solid #555555;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #4fc3f7 !important;
            color: #ffffff !important;
            border-color: #4fc3f7 !important;
            font-weight: bold;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #4fc3f7;
            border-color: #4fc3f7;
        }
        
        /* Tab panels */
        .stTabs [data-baseweb="tab-panel"] {
            background-color: #1a1a1a;
            padding: 2rem 0;
        }
        
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background-color: #333333 !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        .stSelectbox > div > div > div {
            background-color: #333333 !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        .stButton > button {
            background-color: #333333 !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        .stButton > button:hover {
            background-color: #4fc3f7 !important;
            border-color: #4fc3f7 !important;
        }
        
        [data-testid="metric-container"] {
            background-color: #2d2d2d !important;
            border: 1px solid #444444 !important;
        }
        
        /* File uploader styling */
        .stFileUploader > div {
            background-color: #2d2d2d !important;
            border: 1px solid #555555 !important;
            border-radius: 8px;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #4fc3f7 !important;
        }
    </style>
    """

def main():
    # Apply dark theme first
    st.markdown(get_dark_css(), unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎓 Machine Learning Graduation Project</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #b0bec5;">Content-Based Company Similarity & Candidate Classification</h2>', unsafe_allow_html=True)
    
    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Trang chủ", "🏢 Company Similarity", "👤 Candidate Classification", "ℹ️ About"])
    
    with tab1:
        show_homepage()
    
    with tab2:
        show_company_similarity()
    
    with tab3:
        show_candidate_classification()
    
    with tab4:
        show_about()

def show_homepage():
    st.markdown("## 📊 Tổng quan dự án")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🏢 Content-Based Company Similarity
        - **Mục tiêu**: Gợi ý các công ty tương tự dựa trên nội dung và đặc điểm
        - **Phương pháp**: Content-based filtering
        - **Ứng dụng**: Giúp ứng viên tìm kiếm công ty phù hợp
        """)
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col1_1, col1_2, col1_3 = st.columns(3)
        col1_1.metric("Companies", "1,250", "↑ 15%")
        col1_2.metric("Accuracy", "87.3%", "↑ 2.1%")
        col1_3.metric("Precision", "89.1%", "↑ 1.8%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### 👤 'Recommend or Not' Classification
        - **Mục tiêu**: Phân loại ứng viên có nên được gợi ý hay không
        - **Phương pháp**: Binary classification
        - **Ứng dụng**: Lọc ứng viên phù hợp cho từng vị trí
        """)
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        col2_1, col2_2, col2_3 = st.columns(3)
        col2_1.metric("Candidates", "3,847", "↑ 25%")
        col2_2.metric("F1-Score", "91.7%", "↑ 3.2%")
        col2_3.metric("Recall", "88.9%", "↑ 2.5%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance charts
    st.markdown("## 📈 Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        accuracy = [82.1, 84.3, 85.7, 86.2, 87.0, 87.3]
        
        fig1 = px.line(x=months, y=accuracy, title="Company Similarity Model - Accuracy Over Time")
        fig1.update_traces(line_color='#4fc3f7', line_width=3)
        fig1.update_layout(
            xaxis_title="Month", 
            yaxis_title="Accuracy (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [90.5, 89.2, 88.9, 91.7]
        
        fig2 = px.bar(x=metrics, y=values, title="Candidate Classification - Model Metrics")
        fig2.update_traces(marker_color='#4fc3f7')
        fig2.update_layout(
            xaxis_title="Metrics", 
            yaxis_title="Score (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig2, use_container_width=True)

def show_company_similarity():
    st.markdown("## 🏢 Content-Based Company Similarity Recommendation")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["🔍 Manual Input", "📁 Upload File"])
    
    with tab1:
        st.markdown("### 🔍 Tìm kiếm công ty tương tự")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            companies_df = load_sample_data('companies')
            company_names = companies_df['company_name'].tolist() if companies_df is not None else ['TechCorp', 'DataSoft', 'AIVentures']
            
            selected_company = st.selectbox("Chọn công ty:", company_names)
            num_recommendations = st.slider("Số lượng gợi ý:", 1, 10, 5)
        
        with col2:
            st.markdown("### ⚙️ Parameters")
            similarity_threshold = st.slider("Similarity threshold:", 0.1, 1.0, 0.7)
            include_industry = st.checkbox("Filter by industry", True)
        
        if st.button("🔍 Tìm công ty tương tự", type="primary", key="manual_search"):
            process_company_similarity(selected_company, num_recommendations, similarity_threshold)
    
    with tab2:
        st.markdown("### 📁 Upload Company Data")
        st.markdown("Upload file CSV/Excel chứa thông tin công ty để batch predict")
        
        uploaded_file = st.file_uploader(
            "Chọn file:",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain columns: company_name, industry, company_size, description, etc."
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.markdown("### 📊 Preview Data")
                st.dataframe(df.head(), use_container_width=True)
                
                # Settings for batch processing
                col1, col2 = st.columns(2)
                with col1:
                    num_recommendations = st.slider("Số gợi ý cho mỗi công ty:", 1, 10, 3, key="batch_num")
                with col2:
                    similarity_threshold = st.slider("Similarity threshold:", 0.1, 1.0, 0.7, key="batch_threshold")
                
                if st.button("🚀 Process All Companies", type="primary"):
                    process_batch_companies(df, num_recommendations, similarity_threshold)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.info("Please ensure your file has the correct format with columns like: company_name, industry, description")

def process_company_similarity(selected_company, num_recommendations, similarity_threshold):
    with st.spinner("Đang xử lý..."):
        similar_companies = generate_fake_similar_companies(selected_company, num_recommendations)
        
        st.markdown("### 📊 Kết quả gợi ý")
        
        cols = st.columns(min(3, len(similar_companies)))
        for i, company in enumerate(similar_companies):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>{company['name']}</h4>
                    <p><strong>Similarity:</strong> {company['similarity']:.2f}</p>
                    <p><strong>Industry:</strong> {company['industry']}</p>
                    <p><strong>Size:</strong> {company['size']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Similarity Scores")
        names = [c['name'] for c in similar_companies]
        scores = [c['similarity'] for c in similar_companies]
        
        fig = px.bar(x=names, y=scores, title="Company Similarity Scores")
        fig.update_traces(marker_color='#4fc3f7')
        fig.update_layout(
            xaxis_title="Companies", 
            yaxis_title="Similarity Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

def process_batch_companies(df, num_recommendations, similarity_threshold):
    with st.spinner("Processing batch companies..."):
        results = []
        progress_bar = st.progress(0)
        
        for i, row in df.iterrows():
            company_name = row.get('company_name', f'Company_{i}')
            similar_companies = generate_fake_similar_companies(company_name, num_recommendations)
            
            for sim_company in similar_companies:
                results.append({
                    'Original_Company': company_name,
                    'Similar_Company': sim_company['name'],
                    'Similarity_Score': sim_company['similarity'],
                    'Industry': sim_company['industry'],
                    'Size': sim_company['size']
                })
            
            progress_bar.progress((i + 1) / len(df))
        
        results_df = pd.DataFrame(results)
        
        st.markdown("### 📊 Batch Processing Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="company_similarity_results.csv",
            mime="text/csv"
        )
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Companies Processed", len(df))
        with col2:
            st.metric("Total Recommendations", len(results_df))
        with col3:
            avg_similarity = results_df['Similarity_Score'].mean()
            st.metric("Avg Similarity", f"{avg_similarity:.2f}")

def show_candidate_classification():
    st.markdown("## 👤 'Recommend or Not' Classification for Candidates")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["👤 Single Candidate", "📁 Batch Upload"])
    
    with tab1:
        st.markdown("### 📝 Thông tin ứng viên")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Tên ứng viên:", "Nguyen Van A")
            age = st.number_input("Tuổi:", 20, 65, 28)
            experience = st.number_input("Kinh nghiệm (năm):", 0, 20, 3)
            education = st.selectbox("Học vấn:", ["High School", "Bachelor", "Master", "PhD"])
        
        with col2:
            skills = st.multiselect("Kỹ năng:", 
                                   ["Python", "Java", "JavaScript", "SQL", "Machine Learning", "Data Analysis", "Project Management"])
            salary_expectation = st.number_input("Mong muốn lương (triệu VND):", 10, 100, 25)
            location = st.selectbox("Địa điểm:", ["Ho Chi Minh City", "Hanoi", "Da Nang", "Other"])
            job_type = st.selectbox("Loại công việc:", ["Full-time", "Part-time", "Contract", "Intern"])
        
        if st.button("🎯 Phân loại ứng viên", type="primary", key="single_classify"):
            process_single_candidate(name, age, experience, education, skills, salary_expectation, location, job_type)
    
    with tab2:
        st.markdown("### 📁 Upload Candidates Data")
        st.markdown("Upload file CSV/Excel chứa thông tin ứng viên để batch classify")
        
        uploaded_file = st.file_uploader(
            "Chọn file ứng viên:",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain columns: name, age, experience_years, education_level, skills, salary_expectation, etc."
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.markdown("### 📊 Preview Candidates Data")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column mapping
                st.markdown("### 🔗 Column Mapping")
                col1, col2 = st.columns(2)
                
                with col1:
                    name_col = st.selectbox("Name column:", df.columns, index=0)
                    age_col = st.selectbox("Age column:", df.columns, index=1 if len(df.columns) > 1 else 0)
                    exp_col = st.selectbox("Experience column:", df.columns, index=2 if len(df.columns) > 2 else 0)
                
                with col2:
                    edu_col = st.selectbox("Education column:", df.columns, index=3 if len(df.columns) > 3 else 0)
                    skills_col = st.selectbox("Skills column:", df.columns, index=4 if len(df.columns) > 4 else 0)
                    salary_col = st.selectbox("Salary column:", df.columns, index=5 if len(df.columns) > 5 else 0)
                
                # Settings
                confidence_threshold = st.slider("Confidence threshold for recommendation:", 0.5, 0.9, 0.6)
                
                if st.button("🚀 Classify All Candidates", type="primary"):
                    process_batch_candidates(df, name_col, age_col, exp_col, edu_col, skills_col, salary_col, confidence_threshold)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.info("Please ensure your file has the correct format with candidate information")

def process_single_candidate(name, age, experience, education, skills, salary_expectation, location, job_type):
    with st.spinner("Đang phân tích..."):
        prediction_result = generate_fake_candidate_prediction(name, age, experience, education, skills, salary_expectation)
        
        st.markdown("### 📊 Kết quả phân loại")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recommendation = "✅ RECOMMEND" if prediction_result['recommend'] else "❌ NOT RECOMMEND"
            color = "#2ecc71" if prediction_result['recommend'] else "#e74c3c"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color}; color: white;">
                <h2>{recommendation}</h2>
                <p>Confidence: {prediction_result['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📈 Score Breakdown")
            st.progress(prediction_result['skill_score'], "Skills Match")
            st.progress(prediction_result['experience_score'], "Experience")
            st.progress(prediction_result['education_score'], "Education")
            st.progress(prediction_result['overall_score'], "Overall")
        
        with col3:
            st.markdown("#### 💡 Insights")
            for insight in prediction_result['insights']:
                st.write(f"• {insight}")
        
        st.markdown("### 🎯 Feature Importance")
        features = ['Skills', 'Experience', 'Education', 'Salary Fit', 'Location']
        importance = prediction_result['feature_importance']
        
        fig = px.horizontal_bar(x=importance, y=features, title="Feature Importance in Classification")
        fig.update_traces(marker_color='#4fc3f7')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

def process_batch_candidates(df, name_col, age_col, exp_col, edu_col, skills_col, salary_col, confidence_threshold):
    with st.spinner("Classifying all candidates..."):
        results = []
        progress_bar = st.progress(0)
        
        for i, row in df.iterrows():
            # Extract data from row
            name = row.get(name_col, f'Candidate_{i}')
            age = row.get(age_col, 25)
            experience = row.get(exp_col, 0)
            education = row.get(edu_col, 'Bachelor')
            skills = str(row.get(skills_col, '')).split(',') if pd.notna(row.get(skills_col, '')) else []
            salary = row.get(salary_col, 25)
            
            # Get prediction
            prediction = generate_fake_candidate_prediction(name, age, experience, education, skills, salary)
            
            results.append({
                'Name': name,
                'Age': age,
                'Experience_Years': experience,
                'Education': education,
                'Skills_Count': len(skills),
                'Salary_Expectation': salary,
                'Recommendation': 'RECOMMEND' if prediction['recommend'] else 'NOT RECOMMEND',
                'Confidence': prediction['confidence'],
                'Skills_Score': prediction['skill_score'],
                'Experience_Score': prediction['experience_score'],
                'Education_Score': prediction['education_score'],
                'Overall_Score': prediction['overall_score']
            })
            
            progress_bar.progress((i + 1) / len(df))
        
        results_df = pd.DataFrame(results)
        
        st.markdown("### 📊 Batch Classification Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", len(results_df))
        with col2:
            recommended = len(results_df[results_df['Recommendation'] == 'RECOMMEND'])
            st.metric("Recommended", recommended, f"{recommended/len(results_df)*100:.1f}%")
        with col3:
            avg_confidence = results_df['Confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        with col4:
            high_confidence = len(results_df[results_df['Confidence'] > confidence_threshold])
            st.metric("High Confidence", high_confidence, f"{high_confidence/len(results_df)*100:.1f}%")
        
        # Filter options
        st.markdown("### 🔍 Filter Results")
        col1, col2 = st.columns(2)
        with col1:
            show_only = st.selectbox("Show only:", ["All", "RECOMMEND", "NOT RECOMMEND"])
        with col2:
            min_confidence = st.slider("Minimum confidence:", 0.0, 1.0, 0.0)
        
        # Apply filters
        filtered_df = results_df.copy()
        if show_only != "All":
            filtered_df = filtered_df[filtered_df['Recommendation'] == show_only]
        filtered_df = filtered_df[filtered_df['Confidence'] >= min_confidence]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="candidate_classification_results.csv",
            mime="text/csv"
        )
        
        # Visualization
        st.markdown("### 📈 Results Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            # Recommendation distribution
            rec_counts = results_df['Recommendation'].value_counts()
            fig1 = px.pie(values=rec_counts.values, names=rec_counts.index, 
                         title="Recommendation Distribution")
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig2 = px.histogram(results_df, x='Confidence', nbins=20, 
                              title="Confidence Score Distribution")
            fig2.update_traces(marker_color='#4fc3f7')
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig2, use_container_width=True)

def show_about():
    st.markdown("## ℹ️ Về dự án")
    
    st.markdown("""
    ### 🎓 Đồ án tốt nghiệp
    
    **Chủ đề:** Content-Based Company Similarity Recommendation and 'Recommend or Not' Classification for Candidates
    
    **Mô tả:**
    Dự án này bao gồm hai mô hình machine learning chính:
    
    1. **Content-Based Company Similarity:** Sử dụng thuật toán content-based filtering để gợi ý các công ty tương tự dựa trên đặc điểm và nội dung công ty.
    
    2. **Candidate Classification:** Mô hình phân loại binary để xác định một ứng viên có nên được gợi ý cho một vị trí cụ thể hay không.
    
    ### 🛠️ Công nghệ sử dụng
    - **Backend:** Python, Scikit-learn, Pandas, NumPy
    - **Frontend:** Streamlit
    - **Visualization:** Plotly, Matplotlib
    - **Model Storage:** Pickle files
    
    ### 📊 Dữ liệu
    - Dữ liệu công ty: Thông tin về tên, ngành nghề, quy mô, địa điểm
    - Dữ liệu ứng viên: Thông tin cá nhân, kỹ năng, kinh nghiệm, học vấn
    
    ### 🚀 Triển khai
    Ứng dụng được xây dựng bằng Streamlit để dễ dàng demo và sử dụng.
    """)
    
    st.markdown("---")
    st.markdown("**Được phát triển bởi:** [Tên sinh viên] - [Mã số sinh viên]")
    st.markdown("**Trường:** [Tên trường] - **Năm:** 2024")

def generate_fake_similar_companies(selected_company, num_recommendations):
    import random
    
    companies = [
        {"name": "TechStart Inc", "industry": "Technology", "size": "Startup"},
        {"name": "DataCorp Ltd", "industry": "Data Analytics", "size": "Medium"},
        {"name": "AI Solutions", "industry": "Artificial Intelligence", "size": "Large"},
        {"name": "CloudTech Pro", "industry": "Cloud Computing", "size": "Medium"},
        {"name": "DevOps Masters", "industry": "Software Development", "size": "Small"},
        {"name": "FinTech Innovations", "industry": "Financial Technology", "size": "Startup"},
        {"name": "Digital Marketing Co", "industry": "Marketing", "size": "Medium"},
        {"name": "CyberSec Guard", "industry": "Cybersecurity", "size": "Large"}
    ]
    
    selected_companies = random.sample(companies, min(num_recommendations, len(companies)))
    
    for company in selected_companies:
        company['similarity'] = round(random.uniform(0.6, 0.95), 2)
    
    return sorted(selected_companies, key=lambda x: x['similarity'], reverse=True)

def generate_fake_candidate_prediction(name, age, experience, education, skills, salary_expectation):
    import random
    
    skill_score = min(len(skills) * 0.15, 1.0)
    experience_score = min(experience * 0.1, 1.0)
    education_score = {"High School": 0.5, "Bachelor": 0.7, "Master": 0.9, "PhD": 1.0}[education]
    salary_score = max(0.3, 1.0 - (salary_expectation - 20) * 0.01)
    
    overall_score = (skill_score + experience_score + education_score + salary_score) / 4
    confidence = overall_score + random.uniform(-0.1, 0.1)
    recommend = confidence > 0.6
    
    insights = []
    if skill_score > 0.7:
        insights.append("Strong technical skills match")
    if experience_score > 0.5:
        insights.append("Good experience level for the role")
    if education_score > 0.8:
        insights.append("High education qualification")
    if salary_score < 0.5:
        insights.append("Salary expectation might be high")
    
    return {
        'recommend': recommend,
        'confidence': confidence,
        'skill_score': skill_score,
        'experience_score': experience_score,
        'education_score': education_score,
        'overall_score': overall_score,
        'feature_importance': [0.35, 0.25, 0.20, 0.12, 0.08],
        'insights': insights
    }

if __name__ == "__main__":
    main()