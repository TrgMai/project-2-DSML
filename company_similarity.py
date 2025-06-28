import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import json

class CompanySimilarityEngine:
    def __init__(self):
        self.models_loaded = False
        self.models = {}
        self.similarity_matrices = {}
        self.company_data = None
        self.vectorizers = {}
        
    def load_models_and_data(self):
        """Load all similarity models and company data"""
        if self.models_loaded:
            return True
            
        try:
            # Load company data
            if os.path.exists('data/companies_with_recommend.csv'):
                self.company_data = pd.read_csv('data/companies_with_recommend.csv')
            elif os.path.exists('model/tfidf/processed_data.xlsx'):
                self.company_data = pd.read_excel('model/tfidf/processed_data.xlsx')
            else:
                st.error("Không tìm thấy file dữ liệu công ty!")
                return False
            
            # Load metadata if available
            if os.path.exists('model/tfidf/metadata.json'):
                with open('model/tfidf/metadata.json', 'r') as f:
                    self.metadata = json.load(f)
            
            # Load similarity models và matrices
            self._load_tfidf_models()
            self._load_bert_embeddings()
            self._load_doc2vec_models()
            self._load_fasttext_models()
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Lỗi khi load models: {str(e)}")
            return False
    
    def _load_tfidf_models(self):
        """Load TF-IDF models and similarity matrices"""
        tfidf_path = 'model/tfidf/'
        
        model_types = [
            'Company Type', 'Company industry', 'Company overview', 
            'Company size', 'Country', 'Overtime Policy', 'Working days'
        ]
        
        for model_type in model_types:
            try:
                # Load cosine similarity matrices
                cosine_file = f"{tfidf_path}cosine_sim_{model_type}.pkl"
                if os.path.exists(cosine_file):
                    import joblib
                    self.similarity_matrices[f'tfidf_{model_type}'] = joblib.load(cosine_file)
                
                # Load TF-IDF matrices
                matrix_file = f"{tfidf_path}matrix_{model_type}.pkl"
                if os.path.exists(matrix_file):
                    import joblib
                    self.models[f'tfidf_matrix_{model_type}'] = joblib.load(matrix_file)
                
                # Load vectorizers
                vectorizer_file = f"{tfidf_path}vectorizer_{model_type}.pkl"
                if os.path.exists(vectorizer_file):
                    import joblib
                    self.vectorizers[f'tfidf_{model_type}'] = joblib.load(vectorizer_file)
                    
            except Exception as e:
                st.warning(f"Không thể load TF-IDF model cho {model_type}: {str(e)}")
    
    def _load_bert_embeddings(self):
        """Load BERT embeddings"""
        bert_path = 'model/bert/'
        
        embedding_files = [
            'vectors_Company Type.npy', 'vectors_Company industry.npy',
            'vectors_Company overview.npy', 'vectors_Company size.npy',
            'vectors_Country.npy', 'vectors_Overtime Policy.npy',
            'vectors_Working days.npy'
        ]
        
        for file_name in embedding_files:
            file_path = f"{bert_path}{file_name}"
            if os.path.exists(file_path):
                embeddings = np.load(file_path)
                model_name = file_name.replace('vectors_', '').replace('.npy', '')
                self.models[f'bert_{model_name}'] = embeddings
    
    def _load_doc2vec_models(self):
        """Load Doc2Vec models and embeddings"""
        doc2vec_path = 'model/doc2vec/'
        
        # Load precomputed embeddings if available
        embedding_files = [
            'vectors_Company Type.npy', 'vectors_Company industry.npy',
            'vectors_Company overview.npy', 'vectors_Company size.npy',
            'vectors_Country.npy', 'vectors_Overtime Policy.npy',
            'vectors_Working days.npy'
        ]
        
        for file_name in embedding_files:
            file_path = f"{doc2vec_path}{file_name}"
            if os.path.exists(file_path):
                embeddings = np.load(file_path)
                model_name = file_name.replace('vectors_', '').replace('.npy', '')
                self.models[f'doc2vec_vectors_{model_name}'] = embeddings
    
    def _load_fasttext_models(self):
        """Load FastText models"""
        fasttext_path = 'model/fasttext/'
        
        # Load precomputed embeddings
        embedding_files = [
            'vectors_Company Type.npy', 'vectors_Company industry.npy',
            'vectors_Company overview.npy', 'vectors_Company size.npy',
            'vectors_Country.npy', 'vectors_Overtime Policy.npy',
            'vectors_Working days.npy'
        ]
        
        for file_name in embedding_files:
            file_path = f"{fasttext_path}{file_name}"
            if os.path.exists(file_path):
                embeddings = np.load(file_path)
                model_name = file_name.replace('vectors_', '').replace('.npy', '')
                self.models[f'fasttext_vectors_{model_name}'] = embeddings

    def get_company_by_id(self, company_id):
        """Get company information by ID/index"""
        if self.company_data is None:
            return None
        
        try:
            if company_id < len(self.company_data):
                return self.company_data.iloc[company_id]
            else:
                return None
        except:
            return None
    
    def find_similar_companies_tfidf(self, company_id, feature_type='Company overview', top_n=5):
        """Find similar companies using TF-IDF"""
        similarity_key = f'tfidf_{feature_type}'
        
        if similarity_key not in self.similarity_matrices:
            st.error(f"Không tìm thấy similarity matrix cho: {similarity_key}")
            return None
        
        similarity_matrix = self.similarity_matrices[similarity_key]
        
        if company_id >= similarity_matrix.shape[0]:
            st.error(f"Company ID {company_id} vượt quá số lượng companies ({similarity_matrix.shape[0]})")
            return None
        
        # Debug: Check similarity matrix properties
        # st.write(f"**Debug TF-IDF similarity matrix:**")
        # st.write(f"Matrix shape: {similarity_matrix.shape}")
        # st.write(f"Matrix type: {type(similarity_matrix)}")
        # st.write(f"Matrix min: {similarity_matrix.min():.6f}")
        # st.write(f"Matrix max: {similarity_matrix.max():.6f}")
        # st.write(f"Diagonal values (first 10): {np.diag(similarity_matrix)[:10]}")
        # st.write(f"Number of 1.0 values: {np.sum(similarity_matrix == 1.0)}")
        # st.write(f"Total matrix values: {similarity_matrix.size}")
        
        # Check if all values are 1.0
        if np.all(similarity_matrix == 1.0):
            st.error("⚠️ CẢNH BÁO: Tất cả values trong similarity matrix đều = 1.0!")
            st.info("Có thể do TF-IDF vectors bị normalize sai hoặc tất cả documents giống nhau")
        
        # Get similarity scores for the target company
        similarity_scores = similarity_matrix[company_id]
        
        # Debug: Check similarity scores
        # st.write(f"**Debug similarity scores for company {company_id}:**")
        # st.write(f"Max similarity: {similarity_scores.max():.6f}")
        # st.write(f"Min similarity: {similarity_scores.min():.6f}")
        # st.write(f"Mean similarity: {similarity_scores.mean():.6f}")
        # st.write(f"Similarity with itself (should be 1.0): {similarity_scores[company_id]:.6f}")
        
        # Get top similar companies (excluding itself)
        # Set similarity with itself to 0 to exclude it
        similarity_scores_copy = similarity_scores.copy()
        similarity_scores_copy[company_id] = 0
        
        # Handle case where many companies have similarity = 1.0
        max_similarity = similarity_scores_copy.max()
        if max_similarity == 1.0:
            # Count how many companies have similarity = 1.0
            perfect_matches = np.sum(similarity_scores_copy == 1.0)
            st.warning(f"⚠️ Có {perfect_matches} companies với similarity = 1.0 (identical TF-IDF vectors)")
            
            # Add small random noise to break ties for companies with similarity = 1.0
            perfect_indices = np.where(similarity_scores_copy == 1.0)[0]
            np.random.seed(42)  # For reproducible results
            noise = np.random.uniform(0.999, 0.9999, len(perfect_indices))
            similarity_scores_copy[perfect_indices] = noise
        
        similar_indices = similarity_scores_copy.argsort()[::-1][:top_n]
        similar_scores = similarity_scores[similar_indices]  # Use original scores for display
        
        # # Debug: Show top scores
        # st.write(f"**Top {top_n} similar companies:**")
        # for i, (idx, score) in enumerate(zip(similar_indices, similar_scores)):
        #     st.write(f"{i+1}. Company {idx}: {score:.6f}")
        
        return similar_indices, similar_scores
    
    def find_similar_companies_embeddings(self, company_id, model_type='bert', feature_type='Company overview', top_n=5):
        """Find similar companies using embeddings (BERT, Doc2Vec, FastText)"""
        
        # Fix key mapping for different model types
        if model_type == 'bert':
            model_key = f'bert_{feature_type}'
        elif model_type == 'doc2vec':
            model_key = f'doc2vec_vectors_{feature_type}'
        elif model_type == 'fasttext':
            model_key = f'fasttext_vectors_{feature_type}'
        else:
            model_key = f'{model_type}_{feature_type}'
        
        # Debug: Show what key we're looking for
        # st.write(f"**Debug:** Looking for model key: `{model_key}`")
        # st.write(f"**Available keys:** {list(self.models.keys())}")
        
        if model_key not in self.models:
            st.error(f"Model key '{model_key}' not found in loaded models")
            return None
        
        embeddings = self.models[model_key]
        
        if company_id >= embeddings.shape[0]:
            st.error(f"Company ID {company_id} >= embeddings shape {embeddings.shape[0]}")
            return None
        
        # Calculate cosine similarity
        target_embedding = embeddings[company_id].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, embeddings)[0]
        
        # Get top similar companies (excluding itself)
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        similar_scores = similarities[similar_indices]
        
        return similar_indices, similar_scores
    
    def get_available_models(self):
        """Get list of available models and features"""
        available = {
            'tfidf': [],
            'bert': [],
            'doc2vec': [],
            'fasttext': []
        }
        
        for key in self.models.keys():
            if key.startswith('tfidf_matrix_'):
                feature = key.replace('tfidf_matrix_', '')
                available['tfidf'].append(feature)
            elif key.startswith('bert_'):
                feature = key.replace('bert_', '')
                available['bert'].append(feature)
            elif key.startswith('doc2vec_vectors_'):
                feature = key.replace('doc2vec_vectors_', '')
                available['doc2vec'].append(feature)
            elif key.startswith('fasttext_vectors_'):
                feature = key.replace('fasttext_vectors_', '')
                available['fasttext'].append(feature)
        
        return available

def show_company_similarity():
    """Main function cho Candidate Classification tab"""
    st.markdown("## 🏢 Content-Based Company Similarity Recommendation")
    
    # Sidebar cho Company Similarity
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("""
        <div style="background: #2c3e50; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h4 style="color: white; text-align: center; margin-bottom: 1rem;">📊 Menu</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        if st.button("📊 Tổng quan", key="company_overview", use_container_width=True):
            st.session_state.company_page = "overview"
            
        if st.button("🔍 Tìm kiếm công ty", key="company_search", use_container_width=True):
            st.session_state.company_page = "search"
            
        if st.button("📝 Tìm bằng mô tả", key="company_predict", use_container_width=True):
            st.session_state.company_page = "predict"
    
    with col2:
        # Initialize session state
        if 'company_page' not in st.session_state:
            st.session_state.company_page = "overview"
        
        # Content area based on selected page
        if st.session_state.company_page == "overview":
            show_company_overview()
        elif st.session_state.company_page == "search":
            show_search_by_id()
        elif st.session_state.company_page == "predict":
            show_company_predict()

def show_company_overview():
    """Hiển thị tổng quan về hệ thống similarity"""
    st.markdown("### 📊 Tổng quan hệ thống Company Similarity")
    
    # Data pipeline overview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="color: #1976d2; text-align: center;">🧹 Cleaning</h4>
            <ul style="color: #424242; font-size: 14px;">
                <li>Kiểm tra dòng bị thiếu dữ liệu</li>
                <li>Loại bỏ HTML tags, URL, email, số điện thoại</li>
                <li>Chuẩn hóa dấu câu, ký tự đặc biệt, khoảng trắng</li>
                <li>Underthesea để xử lý tiếng Việt</li>
                <li>NLTK để xử lý tiếng Anh</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f3e5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="color: #7b1fa2; text-align: center;">🔧 Prepare</h4>
            <div style="color: #424242; font-size: 14px;">
                <p><strong>Tạo cột mới full_text:</strong></p>
                <ul>
                    <li>'Company overview'</li>
                    <li>'Our key skills'</li>
                    <li>'Why you'll love working here'</li>
                    <li>'Company industry'</li>
                    <li>'Company size'</li>
                    <li>'Company Type'</li>
                    <li>'Location'</li>
                    <li>'Working days'</li>
                    <li>'Overtime Policy'</li>
                    <li>'Country'</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data statistics
    st.markdown("### 📈 Thống kê dữ liệu")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tổng số công ty", "478")
    with col2:
        st.metric("Số features", "9")
    with col3:
        st.metric("Độ dài TB (ký tự)", "825.54")
    with col4:
        st.metric("Độ dài TB (từ)", "130.95")
    
    # Word clouds
    st.markdown("### ☁️ Word Cloud Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Word Cloud cho 'full_text'")
        if os.path.exists('images/wordcloud_full_text.png'):
            st.image('images/wordcloud_full_text.png', width=800)
        else:
            # Create placeholder word cloud data
            st.info("Word Cloud: Các từ phổ biến: 'công ty', 'phát triển', 'công nghệ', 'team', 'work', 'company', 'development', 'technology'")
    
    with col2:
        st.markdown("#### Word Cloud cho 'processed_description'")
        if os.path.exists('images/wordcloud_processed_description.png'):
            st.image('images/wordcloud_processed_description.png', width=800)
        else:
            st.info("Word Cloud: Các từ sau xử lý: 'solution', 'work', 'company', 'team', 'business', 'employee', 'service'")
    
    # Text length distribution
    st.markdown("### 📊 Phân phối độ dài văn bản")
    
    # Generate sample data for text length distribution
    import numpy as np
    
    # Sample data based on the statistics shown
    word_lengths = np.random.gamma(2, 400, 478)  # Approximating the distribution
    char_lengths = word_lengths * 6.3  # Average characters per word
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_words = px.histogram(
            x=word_lengths,
            nbins=30,
            title="Phân phối độ dài văn bản (ký tự)",
            labels={'x': 'Số ký tự', 'y': 'Tần suất'},
            color_discrete_sequence=['#5b9bd5']
        )
        fig_words.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig_words, use_container_width=True)
    
    with col2:
        fig_chars = px.histogram(
            x=char_lengths,
            nbins=30,
            title="Phân phối số từ",
            labels={'x': 'Số từ', 'y': 'Tần suất'},
            color_discrete_sequence=['#5b9bd5']
        )
        fig_chars.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig_chars, use_container_width=True)
    
    # Models information
    st.markdown("### 🤖 Models Overview")
    
    # Models used
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; color: white;">
            <h3>Models</h3>
            <ol style="text-align: left; font-size: 18px; line-height: 2;">
                <li>TfidfVectorizer</li>
                <li>Doc2Vec (Gensim)</li>
                <li>FastText (skipgram)</li>
                <li>SentenceTransformer</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Hiệu suất so sánh các mô hình")
    
    # Model performance comparison
    model_scores = {
        'TFIDF': 0.14,
        'DOC2VEC': 0.97,
        'FASTTEXT': 0.99,
        'BERT': 0.58
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart for model performance
        fig_models = px.bar(
            x=list(model_scores.keys()),
            y=list(model_scores.values()),
            title="Điểm tương đồng trung bình của các mô hình",
            labels={'x': 'Mô hình', 'y': 'Điểm trung bình'},
            color=list(model_scores.values()),
            color_continuous_scale='Blues'
        )
        fig_models.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig_models, use_container_width=True)
    
    with col2:
        # Radar chart for model characteristics
        categories = ['Tốc độ', 'Xử lý ngôn ngữ mới', 'Khả năng học ngữ nghĩa', 'Khả năng cập nhật', 'Tài nguyên']
        
        model_characteristics = {
            'TFIDF': [0.9, 0.3, 0.2, 0.8, 0.9],
            'DOC2VEC': [0.7, 0.8, 0.9, 0.6, 0.7], 
            'FASTTEXT': [0.8, 0.9, 0.8, 0.7, 0.8],
            'BERT': [0.4, 0.9, 0.95, 0.5, 0.3]
        }
        
        fig_radar = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (model, values) in enumerate(model_characteristics.items()):
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model,
                line_color=colors[i]
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Đánh giá đặc tính mô hình",
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Model advantages and disadvantages
    st.markdown("### ✅ Ưu & Nhược điểm của các mô hình:")
    
    model_analysis = [
        {
            'model': 'FASTTEXT',
            'score': 1.0000,
            'advantage': 'Hiệu quả từ mới qua subword',
            'warning': '⚠️ Có thể nhiều nếu văn bản ngắn.'
        },
        {
            'model': 'DOC2VEC', 
            'score': 0.9710,
            'advantage': 'Bắt được ngữ nghĩa tốt hơn TF-IDF',
            'warning': '⚠️ Chất lượng phụ thuộc dữ liệu huấn luyện.'
        },
        {
            'model': 'BERT',
            'score': 0.5886, 
            'advantage': 'Hiểu ngữ cảnh rất tốt',
            'warning': '⚠️ Chậm và tốn tài nguyên hơn.'
        },
        {
            'model': 'TFIDF',
            'score': 0.1411,
            'advantage': 'Đơn giản, nhanh',
            'warning': '⚠️ Phụ thuộc từ khóa, không hiểu ngữ cảnh.'
        }
    ]
    
    for analysis in model_analysis:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.metric("Score", f"{analysis['score']:.4f}")
        
        with col2:
            st.success(f"✅ **{analysis['model']}**: {analysis['advantage']}")
            st.warning(analysis['warning'])
        
        with col3:
            # Visual score representation
            score_percentage = analysis['score'] * 100
            st.progress(analysis['score'], f"{score_percentage:.1f}%")
    
    # Footer with key insights
    st.markdown("---")
    st.markdown("""
    ### 🎯 Kết luận chính:
    
    - **FASTTEXT** cho kết quả tương đồng cao nhất với khả năng xử lý từ mới
    - **DOC2VEC** cân bằng tốt giữa chất lượng và hiệu suất  
    - **BERT** hiểu ngữ cảnh sâu nhưng cần tài nguyên nhiều
    - **TF-IDF** phù hợp cho tìm kiếm nhanh dựa trên từ khóa
    
    Lựa chọn model phụ thuộc vào yêu cầu cụ thể về độ chính xác, tốc độ và tài nguyên.
    """)
    
def show_company_predict():
    """Hiển thị phần tìm kiếm bằng mô tả công ty"""
    st.markdown("### 📝 Tìm kiếm công ty tương tự bằng mô tả")
    
    # Initialize similarity engine
    if 'similarity_engine' not in st.session_state:
        st.session_state.similarity_engine = CompanySimilarityEngine()
    
    engine = st.session_state.similarity_engine
    
    # Load models
    with st.spinner("Đang tải models và dữ liệu..."):
        if not engine.load_models_and_data():
            st.error("Không thể tải models! Vui lòng kiểm tra thư mục models/")
            return
    
    st.success("✅ Đã tải thành công tất cả models!")
    
    # Get available models
    available_models = engine.get_available_models()
    
    # Input form
    st.markdown("#### 📋 Nhập thông tin công ty")
    
    # Company description input
    company_description = st.text_area(
        "Mô tả công ty:",
        placeholder="Ví dụ: Công ty công nghệ chuyên phát triển phần mềm, ứng dụng mobile và AI...",
        height=100,
        help="Nhập mô tả chi tiết về công ty để tìm kiếm các công ty tương tự"
    )
    
    # Additional filters in single row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        company_type_filter = st.selectbox(
            "Loại công ty:",
            ["Tất cả"] + list(set([
                "IT Product", "IT Service", "IT Outsourcing", "Non-IT", 
                "Startup", "Bank", "Unknown"
            ])),
            help="Lọc theo loại công ty"
        )
        
    with col2:
        company_size_filter = st.selectbox(
            "Quy mô công ty:",
            ["Tất cả"] + list(set([
                "1-50 employees", "51-150 employees", "151-300 employees",
                "301+ employees", "Unknown"
            ])),
            help="Lọc theo quy mô công ty"
        )
    
    with col3:
        # Model selection for text similarity
        model_key_mapping = {
            "TF-IDF": "tfidf",
            "BERT": "bert", 
            "DOC2VEC": "doc2vec",
            "FASTTEXT": "fasttext"
        }
        
        selected_model_type = st.selectbox(
            "Model similarity:",
            ["TF-IDF", "BERT", "DOC2VEC", "FASTTEXT"],
            help="Chọn model để tính độ tương tự"
        )
        
        model_key = model_key_mapping[selected_model_type]
    
    with col4:
        # Feature selection
        if model_key in available_models and available_models[model_key]:
            available_features = available_models[model_key]
            selected_feature = st.selectbox(
                "Đặc trưng:",
                available_features,
                index=available_features.index("Company overview") if "Company overview" in available_features else 0,
                help="Chọn đặc trưng để so sánh"
            )
        else:
            st.warning(f"Model {selected_model_type} không có features khả dụng")
            return
    
    # Controls in single row
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    
    with col_ctrl1:
        # Number of results
        top_n = st.slider("Số kết quả:", 3, 20, 10)
    
    with col_ctrl2:
        # Search button
        search_button = st.button(
            "🔍 Tìm kiếm", 
            type="primary",
            use_container_width=True,
            key="company_search_by_description"
        )
    
    # Perform search
    if search_button:
        if not company_description.strip():
            st.warning("⚠️ Vui lòng nhập mô tả công ty!")
            return
            
        search_by_description(
            engine, company_description, model_key, selected_feature, 
            company_type_filter, company_size_filter, top_n
        )

def show_search_by_id():
    """Chỉ cho phần 'Tìm theo ID' trong tab Company Similarity"""
    st.markdown("### 🔍 Tìm kiếm công ty tương tự theo ID")
    
    # Initialize similarity engine
    if 'similarity_engine' not in st.session_state:
        st.session_state.similarity_engine = CompanySimilarityEngine()
    
    engine = st.session_state.similarity_engine
    
    # Load models
    with st.spinner("Đang tải models và dữ liệu..."):
        if not engine.load_models_and_data():
            st.error("Không thể tải models! Vui lòng kiểm tra thư mục models/")
            return
    
    st.success("✅ Đã tải thành công tất cả models!")
    
    # Show available models
    available_models = engine.get_available_models()
    
    # Interface for similarity search
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Company ID input
        total_companies = len(engine.company_data) if engine.company_data is not None else 0
        company_id = st.number_input(
            f"ID Công ty (0-{total_companies-1}):", 
            min_value=0, 
            max_value=max(0, total_companies-1), 
            value=0,
            help=f"Nhập ID từ 0 đến {total_companies-1}"
        )
    
    with col2:
        # Model selection
        selected_model_type = st.selectbox(
            "Loại Model:",
            ["TF-IDF", "BERT", "DOC2VEC", "FASTTEXT"],
            help="Chọn loại model để tính similarity"
        )
    
    with col3:
        # Feature selection - Fix key mapping
        model_key_mapping = {
            "TF-IDF": "tfidf",
            "BERT": "bert", 
            "DOC2VEC": "doc2vec",
            "FASTTEXT": "fasttext"
        }
        
        model_key = model_key_mapping[selected_model_type]
        
        if model_key in available_models and available_models[model_key]:
            available_features = available_models[model_key]
            selected_feature = st.selectbox(
                "Đặc trưng:",
                available_features,
                help="Chọn đặc trưng để so sánh"
            )
        else:
            st.warning(f"Model {selected_model_type} không có features khả dụng")
            return
    
    # Number of similar companies
    if 'selected_feature' in locals():
        top_n = st.slider("Số công ty tương tự:", 3, 20, 5)
        
        # Search button
        if st.button("🔍 Tìm kiếm công ty tương tự", type="primary"):
            search_similar_companies(engine, company_id, model_key, selected_feature, top_n)
    else:
        st.info("Vui lòng chọn model và feature hợp lệ để tiếp tục")

def search_similar_companies(engine, company_id, model_type, feature_type, top_n):
    """Search and display similar companies"""
    
    # Get target company info
    target_company = engine.get_company_by_id(company_id)
    
    if target_company is None:
        st.error(f"Không tìm thấy công ty với ID {company_id}")
        return
    
    # Display target company
    st.markdown("### 🎯 Công ty mục tiêu")
    display_company_card(target_company, company_id, is_target=True)
    
    # Find similar companies
    with st.spinner("Đang tìm kiếm công ty tương tự..."):
        if model_type == 'tfidf':
            result = engine.find_similar_companies_tfidf(company_id, feature_type, top_n)
        else:
            result = engine.find_similar_companies_embeddings(company_id, model_type, feature_type, top_n)
    
    if result is None:
        st.error(f"Không thể thực hiện tìm kiếm với model {model_type} và feature {feature_type}")
        return
    
    similar_indices, similar_scores = result
    
    # Display results
    st.markdown(f"### 🔗 Top {len(similar_indices)} công ty tương tự")
    st.markdown(f"**Model:** {model_type.upper()} | **Feature:** {feature_type}")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Danh sách", "📊 Biểu đồ", "🗺️ Visualization"])
    
    with tab1:
        display_similar_companies_list(engine, similar_indices, similar_scores)
    
    with tab2:
        display_similarity_chart(engine, similar_indices, similar_scores, company_id)
    
    with tab3:
        display_similarity_map(engine, similar_indices, similar_scores, company_id, model_type, feature_type)

def display_company_card(company_data, company_id, is_target=False):
    """Display company information in a card format"""
    
    # Find company name column
    name_col = None
    for col in ['Company Name', 'company_name', 'CompanyName', 'name', 'Name']:
        if col in company_data.index and pd.notna(company_data.get(col)):
            name_col = col
            break
    
    company_name = company_data.get(name_col, f"Company {company_id}") if name_col else f"Company {company_id}"
    
    # Color for target vs similar
    bg_color = "#4fc3f7" if is_target else "#2d2d2d"
    border_color = "#4fc3f7" if is_target else "#555555"
    
    # Create company card
    card_content = f"""
    <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; 
                border: 2px solid {border_color}; margin: 10px 0; color: white;">
        <h4 style="margin: 0; color: white;">{'🎯 ' if is_target else ''}{company_name}</h4>
        <p style="margin: 5px 0; color: #cccccc;"><strong>ID:</strong> {company_id}</p>
    """
    
    # Add key information
    key_fields = [
        'Company Type', 'Company industry', 'Company size', 
        'Country', 'Location', 'Overtime Policy'
    ]
    
    for field in key_fields:
        if field in company_data.index and pd.notna(company_data.get(field)):
            value = company_data[field]
            card_content += f'<p style="margin: 5px 0; color: #cccccc;"><strong>{field}:</strong> {value}</p>'
    
    card_content += "</div>"
    
    st.markdown(card_content, unsafe_allow_html=True)

def display_similar_companies_list(engine, similar_indices, similar_scores):
    """Display list of similar companies"""
    
    for i, (idx, score) in enumerate(zip(similar_indices, similar_scores)):
        company = engine.get_company_by_id(idx)
        if company is not None:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                display_company_card(company, idx)
            
            with col2:
                st.metric(
                    "Similarity", 
                    f"{score:.3f}",
                    help=f"Cosine similarity score: {score:.6f}"
                )

def display_similarity_chart(engine, similar_indices, similar_scores, target_id):
    """Display similarity scores in a chart"""
    
    # Get company names
    company_names = []
    for idx in similar_indices:
        company = engine.get_company_by_id(idx)
        if company is not None:
            # Try to find company name
            name_col = None
            for col in ['Company Name', 'company_name', 'CompanyName', 'name', 'Name']:
                if col in company.index and pd.notna(company.get(col)):
                    name_col = col
                    break
            name = company.get(name_col, f"Company {idx}") if name_col else f"Company {idx}"
            company_names.append(f"{name} (ID: {idx})")
        else:
            company_names.append(f"Company {idx}")
    
    # Create bar chart
    fig = px.bar(
        x=similar_scores,
        y=company_names,
        orientation='h',
        title=f"Similarity Scores vs Company {target_id}",
        labels={'x': 'Similarity Score', 'y': 'Companies'},
        color=similar_scores,
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def search_by_description(engine, description, model_type, feature_type, type_filter, size_filter, top_n):
    """Search for similar companies based on text description"""
    
    with st.spinner("Đang phân tích mô tả và tìm kiếm..."):
        try:
            # Get the appropriate vectorizer and similarity approach
            if model_type == 'tfidf':
                # Use TF-IDF vectorizer to transform input text
                vectorizer_key = f'tfidf_{feature_type}'
                
                if vectorizer_key not in engine.vectorizers:
                    st.error(f"Không tìm thấy vectorizer cho {feature_type}")
                    return
                
                vectorizer = engine.vectorizers[vectorizer_key]
                
                # Transform input description
                input_vector = vectorizer.transform([description])
                
                # Get TF-IDF matrix for comparison
                tfidf_matrix_key = f'tfidf_matrix_{feature_type}'
                if tfidf_matrix_key not in engine.models:
                    st.error(f"Không tìm thấy TF-IDF matrix cho {feature_type}")
                    return
                
                tfidf_matrix = engine.models[tfidf_matrix_key]
                
                # Calculate cosine similarity
                similarities = cosine_similarity(input_vector, tfidf_matrix)[0]
                
            else:
                # For embedding models (BERT, Doc2Vec, FastText)
                st.warning("⚠️ Tìm kiếm bằng text description chỉ hỗ trợ TF-IDF model hiện tại")
                st.info("Các embedding models (BERT, Doc2Vec, FastText) cần pre-trained model để encode text mới")
                return
            
            # Get company data for filtering
            company_data = engine.company_data
            
            # Apply filters if specified
            valid_indices = list(range(len(company_data)))
            
            if type_filter != "Tất cả":
                type_col = None
                for col in ['Company Type', 'company_type', 'CompanyType']:
                    if col in company_data.columns:
                        type_col = col
                        break
                
                if type_col:
                    type_mask = company_data[type_col] == type_filter
                    valid_indices = [i for i in valid_indices if type_mask.iloc[i]]
            
            if size_filter != "Tất cả":
                size_col = None
                for col in ['Company size', 'company_size', 'CompanySize']:
                    if col in company_data.columns:
                        size_col = col
                        break
                
                if size_col:
                    size_mask = company_data[size_col] == size_filter
                    valid_indices = [i for i in valid_indices if size_mask.iloc[i]]
            
            # Filter similarities by valid indices
            filtered_similarities = [(i, similarities[i]) for i in valid_indices]
            
            # Sort by similarity score
            filtered_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            top_results = filtered_similarities[:top_n]
            
            if not top_results:
                st.warning("Không tìm thấy công ty nào phù hợp với bộ lọc đã chọn")
                return
            
            # Display results
            st.markdown("### 🎯 Kết quả tìm kiếm")
            st.markdown(f"**Mô tả tìm kiếm:** {description[:100]}{'...' if len(description) > 100 else ''}")
            st.markdown(f"**Model:** {model_type.upper()} | **Feature:** {feature_type}")
            
            if type_filter != "Tất cả" or size_filter != "Tất cả":
                filters_applied = []
                if type_filter != "Tất cả":
                    filters_applied.append(f"Loại: {type_filter}")
                if size_filter != "Tất cả":
                    filters_applied.append(f"Quy mô: {size_filter}")
                st.markdown(f"**Bộ lọc:** {', '.join(filters_applied)}")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📋 Danh sách kết quả", "📊 Biểu đồ similarity"])
            
            with tab1:
                display_search_results_list(engine, top_results)
            
            with tab2:
                display_search_results_chart(engine, top_results, description)
                
        except Exception as e:
            st.error(f"Lỗi trong quá trình tìm kiếm: {str(e)}")
            st.code(str(e))

def display_search_results_list(engine, results):
    """Display search results in list format"""
    
    for i, (company_idx, similarity_score) in enumerate(results):
        company = engine.get_company_by_id(company_idx)
        if company is not None:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Display company card with ranking
                display_company_card_with_rank(company, company_idx, i+1)
            
            with col2:
                st.metric(
                    "Similarity", 
                    f"{similarity_score:.3f}",
                    help=f"Similarity score: {similarity_score:.6f}"
                )
                
                # Add similarity bar
                similarity_percentage = similarity_score * 100
                st.progress(similarity_score, f"{similarity_percentage:.1f}%")

def display_company_card_with_rank(company_data, company_id, rank):
    """Display company information card with ranking"""
    
    # Find company name column
    name_col = None
    for col in ['Company Name', 'company_name', 'CompanyName', 'name', 'Name']:
        if col in company_data.index and pd.notna(company_data.get(col)):
            name_col = col
            break
    
    company_name = company_data.get(name_col, f"Company {company_id}") if name_col else f"Company {company_id}"
    
    # Rank-based color
    if rank <= 3:
        bg_color = "#2ecc71"  # Green for top 3
    elif rank <= 7:
        bg_color = "#f39c12"  # Orange for top 7
    else:
        bg_color = "#3498db"  # Blue for others
    
    # Create company card
    card_content = f"""
    <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; 
                border: 2px solid #ffffff; margin: 10px 0; color: white;">
        <h4 style="margin: 0; color: white;">#{rank} {company_name}</h4>
        <p style="margin: 5px 0; color: #cccccc;"><strong>ID:</strong> {company_id}</p>
    """
    
    # Add key information
    key_fields = [
        'Company Type', 'Company industry', 'Company size', 
        'Country', 'Location', 'Overtime Policy'
    ]
    
    for field in key_fields:
        if field in company_data.index and pd.notna(company_data.get(field)):
            value = company_data[field]
            card_content += f'<p style="margin: 5px 0; color: #cccccc;"><strong>{field}:</strong> {value}</p>'
    
    # Add description if available
    for desc_field in ['Company overview', 'company_overview', 'description', 'Description']:
        if desc_field in company_data.index and pd.notna(company_data.get(desc_field)):
            description = str(company_data[desc_field])
            if len(description) > 150:
                description = description[:150] + "..."
            card_content += f'<p style="margin: 5px 0; color: #cccccc; font-style: italic;"><strong>Mô tả:</strong> {description}</p>'
            break
    
    card_content += "</div>"
    
    st.markdown(card_content, unsafe_allow_html=True)

def display_search_results_chart(engine, results, query_description):
    """Display search results in chart format"""
    
    if not results:
        st.warning("Không có dữ liệu để hiển thị biểu đồ")
        return
    
    # Prepare data for chart
    company_names = []
    similarity_scores = []
    
    for company_idx, score in results:
        company = engine.get_company_by_id(company_idx)
        if company is not None:
            # Get company name
            name_col = None
            for col in ['Company Name', 'company_name', 'CompanyName', 'name', 'Name']:
                if col in company.index and pd.notna(company.get(col)):
                    name_col = col
                    break
            name = company.get(name_col, f"Company {company_idx}") if name_col else f"Company {company_idx}"
            company_names.append(f"{name} (ID: {company_idx})")
            similarity_scores.append(score)
    
    # Create bar chart
    fig = px.bar(
        x=similarity_scores,
        y=company_names,
        orientation='h',
        title=f"Company Similarity Scores vs Input Description",
        labels={'x': 'Similarity Score', 'y': 'Companies'},
        color=similarity_scores,
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=max(400, len(results) * 40),  # Dynamic height based on number of results
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Similarity cao nhất", f"{max(similarity_scores):.3f}")
    with col2:
        st.metric("Similarity trung bình", f"{np.mean(similarity_scores):.3f}")
def display_similarity_map(engine, similar_indices, similar_scores, target_id, model_type, feature_type):
    """Display similarity visualization using PCA"""
    
    try:
        # Get embeddings for visualization
        if model_type == 'tfidf':
            model_key = f'tfidf_matrix_{feature_type}'
            if model_key in engine.models:
                embeddings = engine.models[model_key].toarray()
            else:
                st.warning("Không có dữ liệu để visualize cho TF-IDF")
                return
        else:
            model_key = f'{model_type}_vectors_{feature_type}' if 'vectors' not in model_type else f'{model_type}_{feature_type}'
            if model_key in engine.models:
                embeddings = engine.models[model_key]
            else:
                st.warning(f"Không có dữ liệu để visualize cho {model_type}")
                return
        
        # Select subset for visualization (target + similar + random sample)
        all_indices = [target_id] + list(similar_indices)
        
        # Add some random companies for context
        total_companies = embeddings.shape[0]
        random_indices = np.random.choice(
            [i for i in range(total_companies) if i not in all_indices], 
            size=min(20, total_companies - len(all_indices)), 
            replace=False
        )
        all_indices.extend(random_indices)
        
        # Get embeddings for selected companies
        selected_embeddings = embeddings[all_indices]
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(selected_embeddings)
        
        # Prepare data for plotting
        plot_data = []
        for i, idx in enumerate(all_indices):
            company = engine.get_company_by_id(idx)
            
            # Get company name
            name_col = None
            for col in ['Company Name', 'company_name', 'CompanyName', 'name', 'Name']:
                if col in company.index and pd.notna(company.get(col)):
                    name_col = col
                    break
            name = company.get(name_col, f"Company {idx}") if name_col else f"Company {idx}"
            
            # Determine point type
            if idx == target_id:
                point_type = "Target"
                size = 20
                color = "#ff6b6b"
            elif idx in similar_indices:
                point_type = "Similar"
                size = 15
                color = "#4ecdc4"
            else:
                point_type = "Other"
                size = 8
                color = "#95a5a6"
            
            plot_data.append({
                'x': embeddings_2d[i, 0],
                'y': embeddings_2d[i, 1],
                'company_name': name,
                'company_id': idx,
                'type': point_type,
                'size': size,
                'color': color
            })
        
        # Create scatter plot
        df_plot = pd.DataFrame(plot_data)
        
        fig = px.scatter(
            df_plot, 
            x='x', 
            y='y',
            color='type',
            size='size',
            hover_data=['company_name', 'company_id'],
            title=f"Company Similarity Visualization - {model_type.upper()} ({feature_type})",
            color_discrete_map={
                "Target": "#ff6b6b",
                "Similar": "#4ecdc4", 
                "Other": "#95a5a6"
            }
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=600,
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show explained variance
        st.info(f"PCA giải thích {pca.explained_variance_ratio_.sum():.1%} variance của dữ liệu gốc")
        
    except Exception as e:
        st.error(f"Lỗi khi tạo visualization: {str(e)}")
        st.info("Có thể do dữ liệu embeddings không tương thích hoặc thiếu dependencies")