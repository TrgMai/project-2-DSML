import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os

from candidate_classification import candidate_classification_tab
from company_similarity import show_company_similarity

# Cấu hình trang
st.set_page_config(
    page_title="ITViec - Data Science and Machine Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS tùy chỉnh để tạo giao diện giống ITViec
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .itviec-logo {
        background: #e74c3c;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        display: inline-block;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        color: #bdc3c7;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .metric-container {
        background: #2c3e50;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-title {
        color: #95a5a6;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: white;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .tab-style {
        background: #34495e;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.2rem;
        color: white;
        border: none;
    }
    
    .active-tab {
        background: #3498db;
    }
    
    .stSelectbox > div > div {
        background-color: #34495e;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .recommendation-card {
        background: #34495e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #e74c3c;
    }
    
    .company-card {
        background: #2c3e50;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header chính
if os.path.exists("images/banner.png"):
        st.image("images/banner.png", use_container_width=True)
st.markdown("""
<div class="main-header">
    <div class="main-title">Đồ án tốt nghiệp - Data Science and Machine Learning</div>
    <div class="feature-title">Classification for Candidates</div>
    <div style="margin-top: 1rem;">
        <div class="feature-title">Content-Based Company Similarity & Recommendation and "Recommend or Not"</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tạo tabs điều hướng
tab1, tab2, tab3 = st.tabs(["🔍 Tổng quan", "👥 Company Similarity", "🎯 Candidate Classification"])

with tab1:
    st.markdown("### 📊 Tổng quan hệ thống")
    
    # Metrics chính
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-title">Tổng công ty</div>
            <div class="metric-value">478</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-title">Tổng ứng viên</div>
            <div class="metric-value">8,417</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-title">Độ chính xác AI</div>
            <div class="metric-value">89.5%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-title">Thời gian xử lý</div>
            <div class="metric-value">0.3s</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Biểu đồ phân bố mới
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏢 Phân phối theo loại công ty")
        company_types = ['IT Product', 'IT Outsourcing', 'IT Service and IT Consulting', 'Non-IT']
        company_values = [295, 86, 68, 29]
        
        fig = px.pie(values=company_values, names=company_types, 
                    title="Phân bố theo loại công ty",
                    color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12', '#95a5a6'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 👥 Phân phối theo quy mô công ty")
        company_sizes = ['1-50 employees', '51-150 employees', '1000+ employees', 
                        '151-300 employees', '301-500 employees', '501-1000 employees']
        size_values = [178, 138, 54, 51, 33, 24]
        
        fig = go.Figure(data=[go.Bar(x=company_sizes, y=size_values,
                               marker_color=['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c', '#34495e'])])
        
        fig.update_layout(
            title="Phân bố theo quy mô công ty",
            xaxis_title="Quy mô nhân viên",
            yaxis_title="Số lượng công ty",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### ⏰ Chính sách làm thêm giờ")
        ot_policies = ['No OT', 'Extra salary for OT', 'Extra days off for OT', 'OT included in base salary']
        ot_values = [389, 52, 5, 1]
        
        fig = go.Figure(data=[go.Bar(y=ot_policies, x=ot_values, orientation='h',
                               marker_color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])])
        
        fig.update_layout(
            title="Chính sách làm thêm giờ",
            xaxis_title="Số lượng công ty",
            yaxis_title="Chính sách OT",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.markdown("#### 🌍 Phân phối theo quốc gia")
        countries = ['Vietnam', 'United States', 'Japan', 'Singapore', 'South Korea', 
                    'Australia', 'France', 'Switzerland', 'Germany', 'United Kingdom']
        country_values = [259, 45, 41, 32, 20, 12, 9, 8, 8, 8]
        
        fig = px.pie(values=country_values, names=countries, 
                    title="Phân bố theo quốc gia",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top 10 ngành nghề
    st.markdown("#### 🏆 Top 10 ngành nghề phổ biến")
    industries = ['Software Products and Web Services', 'IT Services and IT Consulting', 
                 'Software Development Outsourcing', 'Financial Services', 'E-commerce',
                 'AI, Blockchain and Deep Tech Services', 'Banking', 'Game', 'Healthcare',
                 'Media, Advertising and Entertainment']
    industry_values = [107, 104, 66, 30, 23, 18, 14, 11, 10, 8]
    
    fig = go.Figure(data=[go.Bar(x=industries, y=industry_values,
                           marker_color=['#1abc9c', '#3498db', '#9b59b6', '#e74c3c', '#f39c12',
                                       '#2ecc71', '#34495e', '#e67e22', '#95a5a6', '#16a085'])])
    
    fig.update_layout(
        title="Top 10 ngành nghề phổ biến",
        xaxis_title="Ngành nghề",
        yaxis_title="Số lượng công ty",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_tickangle=-45,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    show_company_similarity()
with tab3:
    candidate_classification_tab()

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer-container {
    width: 100%;
    margin: 40px auto;
    padding: 25px;
    border-radius: 12px;
    background-color: #2c3e50;
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
    color: #ecf0f1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.footer-container h4 {
    font-size: 18px;
    font-weight: 600;
}
.footer-container .title {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
    color: #f1c40f;
}
.footer-container a {
    color: #1abc9c;
    text-decoration: none;
}
.footer-container a:hover {
    text-decoration: underline;
}
.footer-container hr {
    margin: 18px auto;
    width: 60%;
    border: 0.5px solid #7f8c8d;
}
.footer-container p {
    margin: 6px 0;
    font-size: 15px;
}
</style>

<div class="footer-container">
    <p class="title">🎓 Đồ án tốt nghiệp – Data Science & Machine Learning</p>
    <h4>Phát triển bởi</h4>
    <p>• <strong>Trần Hoàng Hôn</strong> – <a href="mailto:hoanghonhutech@gmail.com">hoanghonhutech@gmail.com</a></p>
    <p>• <strong>Trương Mai</strong> – <a href="mailto:trgmai98.dev@gmail.com">trgmai98.dev@gmail.com</a></p>
    <hr>
    <p><em>Made with ❤️ using <strong>Streamlit</strong> & <strong>Machine Learning</strong></em></p>
</div>
""", unsafe_allow_html=True)
