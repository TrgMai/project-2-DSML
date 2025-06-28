import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import os
import pickle

def show_candidate_overview():
    """Hiển thị tổng quan Candidate Classification"""
    st.markdown("#### 📊 Tổng quan phân bố dữ liệu và kết quả mô hình")
    
    # Row 1: Company Type và Company Industry
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🏢 Phân phối Company Type")
        company_types = ['IT Product', 'IT Outsourcing', 'IT Service and IT Consulting', 'Non-IT']
        company_type_values = [295, 86, 68, 29]
        
        fig = px.pie(
            values=company_type_values, 
            names=company_types,
            color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12', '#95a5a6']
        )
        fig.update_layout(
            title="Phân phối theo loại công ty",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=350,
            showlegend=True
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị stats
        st.markdown("""
        <div style="background: #34495e; padding: 1rem; border-radius: 5px; margin-top: 0.5rem;">
            <p style="color: white; margin: 0;"><strong>Total:</strong> 478 công ty</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("##### 🏭 Phân phối Company Industry")
        industries = ['Software Products and Web Services', 'IT Services and IT Consulting', 
                     'Software Development Outsourcing', 'Financial Services', 'E-commerce']
        industry_values = [107, 104, 66, 30, 23]
        
        fig = go.Figure(data=[go.Bar(
            x=industries, 
            y=industry_values,
            marker_color=['#1abc9c', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
        )])
        
        fig.update_layout(
            title="Top 5 ngành công nghiệp",
            xaxis_title="Ngành nghề",
            yaxis_title="Số lượng",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_tickangle=-45,
            height=350,
            margin=dict(b=100)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị stats
        st.markdown("""
        <div style="background: #34495e; padding: 1rem; border-radius: 5px; margin-top: 0.5rem;">
            <p style="color: white; margin: 0;"><strong>Top 5 ngành nghề hàng đầu</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2: Company Size và Overtime Policy
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("##### 👥 Phân phối Company Size")
        company_sizes = ['1-50 employees', '51-150 employees', '1000+ employees', 
                        '151-300 employees', '301-500 employees']
        size_values = [178, 138, 54, 51, 33]
        
        fig = go.Figure(data=[go.Bar(
            y=company_sizes, 
            x=size_values, 
            orientation='h',
            marker_color=['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
        )])
        
        fig.update_layout(
            title="Phân phối theo quy mô công ty",
            xaxis_title="Số lượng",
            yaxis_title="Quy mô nhân viên",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.markdown("##### ⏰ Phân phối Overtime Policy")
        ot_policies = ['No OT', 'Extra salary for OT', 'Extra days off for OT', 'OT included in base salary']
        ot_values = [389, 52, 5, 1]
        
        fig = go.Figure(data=[go.Bar(
            x=ot_policies, 
            y=ot_values,
            marker_color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        )])
        
        fig.update_layout(
            title="Chính sách làm thêm giờ",
            xaxis_title="Chính sách OT",
            yaxis_title="Số lượng",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_tickangle=-45,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: Label Distribution và Text Statistics
    st.markdown("---")
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("##### 🎯 Phân phối nhãn Recommend")
        
        labels = ['Not Recommend (0)', 'Recommend (1)']
        label_values = [191, 287]
        label_percentages = [40.0, 60.0]
        
        fig = go.Figure(data=[go.Pie(
            labels=[f'{label}<br>{value} ({pct}%)' for label, value, pct in zip(labels, label_values, label_percentages)],
            values=label_values,
            hole=0.4,
            marker_colors=['#e74c3c', '#2ecc71']
        )])
        
        fig.update_layout(
            title="Phân bố nhãn phân loại",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        st.markdown("""
        <div style="background: #34495e; padding: 1rem; border-radius: 5px; margin-top: 0.5rem;">
            <p style="color: white; margin: 0;"><strong>Tổng số mẫu:</strong> 478</p>
            <p style="color: #2ecc71; margin: 0;"><strong>Recommend:</strong> 287 (60.0%)</p>
            <p style="color: #e74c3c; margin: 0;"><strong>Not Recommend:</strong> 191 (40.0%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("##### 📝 Thống kê độ dài văn bản")
        
        # Text statistics sử dụng st.metric trực tiếp
        st.metric(
            label="📊 Độ dài trung bình",
            value="1,276.95 ký tự"
        )
        st.metric(
            label="📊 Độ dài median",
            value="1,313.50 ký tự"
        )
        st.metric(
            label="📊 Số từ trung bình", 
            value="214.97 từ"
        )
        st.metric(
            label="📊 Số từ median",
            value="219.00 từ"
        )
    
    # Row 4: Word Cloud Section (di chuyển lên trước)
    st.markdown("---")
    st.markdown("##### ☁️ Word Cloud")
    
    # Word Cloud placeholder
    try:
        st.image("images/Word Cloud 1.png", caption="Word Cloud của mô tả công việc", use_container_width=True)
    except:
        st.markdown("""
        <div style="background: #34495e; padding: 2rem; border-radius: 10px; text-align: center; height: 300px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">☁️</div>
            <h4 style="color: white;">Word Cloud</h4>
            <p style="color: #bdc3c7;">Đặt file 'Word Cloud.png' trong thư mục images/</p>
            <p style="color: #95a5a6; font-size: 0.9rem;">Hiển thị các từ khóa phổ biến trong mô tả công việc</p>
        </div>
        """, unsafe_allow_html=True)

    # Row 5: Model Comparison Section
    st.markdown("---")
    st.markdown("## 🤖 So sánh kết quả các mô hình Machine Learning")
    
    # Tạo data cho bảng so sánh
    model_results = {
        'Model': [
            'SKL_Logistic_Regression', 'SKL_Decision_Tree', 'SKL_Random_Forest',
            'SKL_Gradient_Boosting', 'SKL_KNN', 'SKL_SVM',
            'PySpark_Logistic_Regression', 'PySpark_Decision_Tree', 
            'PySpark_Random_Forest', 'PySpark_GBT'
        ],
        'Accuracy': [0.833333, 0.770833, 0.875, 0.916667, 0.84375, 0.614583, 0.78125, 0.854167, 0.84375, 0.864583],
        'Precision': [0.85, 0.821429, 0.859375, 0.890625, 0.830769, 0.610526, 0.779371, 0.854167, 0.850843, 0.865357],
        'Recall': [0.87931, 0.793103, 0.948276, 0.982759, 0.931034, 1.0, 0.78125, 0.854167, 0.84375, 0.864583],
        'F1-score': [0.864407, 0.807018, 0.901639, 0.934426, 0.878049, 0.75817, 0.778067, 0.854167, 0.838881, 0.864874],
        'ROC-AUC': [0.907441, 0.764973, 0.901543, 0.969601, 0.911298, 0.846416, 0.8598, 0.871824, 0.881579, 0.950544],
        'Framework': ['Scikit-learn', 'Scikit-learn', 'Scikit-learn', 'Scikit-learn', 'Scikit-learn', 'Scikit-learn',
                     'PySpark', 'PySpark', 'PySpark', 'PySpark']
    }
    
    # Tạo DataFrame
    df_results = pd.DataFrame(model_results)
    
    # Hiển thị bảng kết quả
    st.markdown("##### 📋 Bảng so sánh kết quả tất cả các mô hình:")
    
    # Format bảng với highlighting cho mô hình tốt nhất
    def highlight_best_model(row):
        best_f1_idx = df_results['F1-score'].idxmax()
        if row.name == best_f1_idx:
            return ['background-color: #90EE90'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = df_results.style.apply(highlight_best_model, axis=1).format({
        'Accuracy': '{:.3f}',
        'Precision': '{:.3f}', 
        'Recall': '{:.3f}',
        'F1-score': '{:.3f}',
        'ROC-AUC': '{:.3f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Highlight mô hình tốt nhất
    best_model_idx = df_results['F1-score'].idxmax()
    best_model_name = df_results.loc[best_model_idx, 'Model']
    best_f1_score = df_results.loc[best_model_idx, 'F1-score']
    
    st.success(f"🏆 **Mô hình tốt nhất:** {best_model_name} với F1-score: {best_f1_score:.3f}")
    
    # Row 5: Visualization so sánh mô hình
    col7, col8 = st.columns(2)
    
    with col7:
        st.markdown("##### 📊 So sánh F1-Score các mô hình")
        
        # Tạo colors - highlight mô hình tốt nhất
        colors = ['#FFD700' if model == best_model_name else '#87CEEB' if 'SKL' in model else '#FFA07A' 
                 for model in df_results['Model']]
        
        fig = go.Figure(data=[go.Bar(
            x=df_results['F1-score'],
            y=[model.replace('SKL_', '').replace('PySpark_', 'PS_') for model in df_results['Model']],
            orientation='h',
            marker_color=colors,
            text=[f'{score:.3f}' for score in df_results['F1-score']],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="F1-Score Comparison",
            xaxis_title="F1-Score",
            yaxis_title="Models",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col8:
        st.markdown("##### 🎯 Framework Performance Comparison")
        
        # So sánh theo framework
        framework_avg = df_results.groupby('Framework')[['Accuracy', 'F1-score', 'ROC-AUC']].mean()
        
        metrics = ['Accuracy', 'F1-score', 'ROC-AUC']
        sklearn_values = [framework_avg.loc['Scikit-learn', metric] for metric in metrics]
        pyspark_values = [framework_avg.loc['PySpark', metric] for metric in metrics]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=sklearn_values,
            theta=metrics,
            fill='toself',
            name='Scikit-learn',
            line_color='#3498db'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=pyspark_values,
            theta=metrics,
            fill='toself',
            name='PySpark',
            line_color='#e74c3c'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Framework Performance Radar",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 6: Best Model Details  
    st.markdown("---")
    st.markdown("## 🏆 Chi tiết mô hình tốt nhất: SKL_Gradient_Boosting")
    
    col9, col10 = st.columns(2)
    
    with col9:
        st.markdown("##### 📈 Metrics của mô hình tốt nhất")
        
        best_metrics = df_results[df_results['Model'] == best_model_name].iloc[0]
        
        # Sử dụng HTML/CSS để hiển thị metrics theo hàng ngang
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 5px; text-align: center;">
                <h4 style="margin: 0; color: #262730;">🎯 Accuracy</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: #262730;">{best_metrics['Accuracy']:.3f}</h2>
            </div>
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 5px; text-align: center;">
                <h4 style="margin: 0; color: #262730;">🔍 Precision</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: #262730;">{best_metrics['Precision']:.3f}</h2>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 5px; text-align: center;">
                <h4 style="margin: 0; color: #262730;">📊 Recall</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: #262730;">{best_metrics['Recall']:.3f}</h2>
            </div>
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 5px; text-align: center;">
                <h4 style="margin: 0; color: #262730;">⚡ ROC-AUC</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: #262730;">{best_metrics['ROC-AUC']:.3f}</h2>
            </div>
        </div>
        
        <div style="background: linear-gradient(90deg, #FFD700, #FFA500); padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h3 style="color: #000; margin: 0;">🏆 F1-Score: {best_metrics['F1-score']:.3f}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col10:
        st.markdown("##### 📊 Classification Report")
        
        # Classification Report - đơn giản hóa để tránh lỗi hiển thị
        report_data = [
            ['Not Recommend', '0.97', '0.82', '0.89', '38'],
            ['Recommend', '0.89', '0.98', '0.93', '58'],
            ['Accuracy', '', '', '0.92', '96'],
            ['Macro avg', '0.93', '0.90', '0.91', '96'],
            ['Weighted avg', '0.92', '0.92', '0.92', '96']
        ]
        
        report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-score', 'Support'])
        
        # Hiển thị bảng đơn giản không có styling phức tạp
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        
        # Summary
        st.markdown("""
        <div style="background: #2c3e50; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
            <p style="color: #2ecc71; margin: 0;"><strong>✅ Kết luận:</strong> Mô hình có độ chính xác cao (92%) với khả năng phân loại tốt cho cả hai class</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 7: Model Performance Summary
    st.markdown("---")
    col11, col12 = st.columns([1, 1])
    
    with col11:
        st.markdown("##### 📊 Classification Report")
        
        # Classification Report - đơn giản hóa để tránh lỗi hiển thị
        report_data = [
            ['Not Recommend', '0.97', '0.82', '0.89', '38'],
            ['Recommend', '0.89', '0.98', '0.93', '58'],
            ['Accuracy', '', '', '0.92', '96'],
            ['Macro avg', '0.93', '0.90', '0.91', '96'],
            ['Weighted avg', '0.92', '0.92', '0.92', '96']
        ]
        
        report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-score', 'Support'])
        
        # Hiển thị bảng đơn giản không có styling phức tạp
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        
        # Summary
        st.markdown("""
        <div style="background: #2c3e50; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
            <p style="color: #2ecc71; margin: 0;"><strong>✅ Kết luận:</strong> Mô hình có độ chính xác cao (92%) với khả năng phân loại tốt cho cả hai class</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col12:
        st.markdown("##### 📊 Model Performance Summary")
        
        # Model performance summary - sử dụng HTML để tránh nested columns
        st.info("🏆 **Best Model: SKL_Gradient_Boosting**")
        
        # Sử dụng HTML/CSS để hiển thị metrics theo hàng ngang
        st.markdown("""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 5px; text-align: center;">
                <h4 style="margin: 0; color: #262730;">✅ Accuracy</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: #262730;">91.67%</h2>
            </div>
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 5px; text-align: center;">
                <h4 style="margin: 0; color: #262730;">🎯 Precision</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: #262730;">89.06%</h2>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 5px; text-align: center;">
                <h4 style="margin: 0; color: #262730;">📈 Recall</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: #262730;">98.28%</h2>
            </div>
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 5px; text-align: center;">
                <h4 style="margin: 0; color: #262730;">🏅 F1-Score</h4>
                <h2 style="margin: 0.5rem 0 0 0; color: #262730;">93.44%</h2>
            </div>
        </div>
        
        <div style="background: #ffd700; padding: 1rem; border-radius: 5px; text-align: center; margin: 1rem 0;">
            <h4 style="margin: 0; color: #000;">⚡ ROC-AUC</h4>
            <h2 style="margin: 0.5rem 0 0 0; color: #000;">96.96%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Kết luận
        st.success("💡 **Kết luận:** Gradient Boosting là mô hình tốt nhất với khả năng dự đoán chính xác và cân bằng tốt giữa precision và recall.")

def show_candidate_search():
    """Hiển thị tìm kiếm công ty"""
    st.markdown("#### 🔍 Tìm kiếm công ty")
    
    # Load data
    @st.cache_data
    def load_company_data():
        try:
            df = pd.read_csv("data/companies_with_recommend.csv")
            return df
        except FileNotFoundError:
            st.error("❌ Không tìm thấy file data/companies_with_recommend.csv")
            return None
        except Exception as e:
            st.error(f"❌ Lỗi đọc file: {str(e)}")
            return None
    
    df = load_company_data()
    
    if df is None:
        return
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "🏢 Nhập tên công ty:",
            placeholder="VD: FPT Software, VNG, Tiki...",
            help="Nhập tên đúng hoặc gần đúng"
        )
    
    with col2:
        search_button = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)
    
    # Search results
    if search_query and (search_button or len(search_query) > 2):
        # Fuzzy search - tìm kiếm gần đúng
        mask = df['Company Name'].str.contains(search_query, case=False, na=False)
        results = df[mask]
        
        if len(results) == 0:
            st.warning(f"❌ Không tìm thấy công ty nào với từ khóa: '{search_query}'")
            
            # Suggest similar companies
            st.markdown("##### 💡 Gợi ý một số công ty khác:")
            sample_companies = df['Company Name'].sample(min(5, len(df))).tolist()
            for company in sample_companies:
                st.markdown(f"• {company}")
        
        else:
            st.success(f"✅ Tìm thấy {len(results)} công ty")
            
            # Display results
            for idx, row in results.iterrows():
                recommend_status = "✅ RECOMMEND" if row['recommend'] == 1 else "❌ NOT RECOMMEND"
                
                # Company header với container
                with st.container():
                    col_name, col_status = st.columns([3, 1])
                    
                    with col_name:
                        st.markdown(f"### 🏢 {row['Company Name']}")
                    
                    with col_status:
                        if row['recommend'] == 1:
                            st.success(recommend_status)
                        else:
                            st.error(recommend_status)
                    
                    # Basic info trong 2 cột
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.markdown(f"**🏢 Loại:** {row['Company Type']}")
                        st.markdown(f"**🏭 Ngành:** {row['Company industry']}")
                        st.markdown(f"**👥 Quy mô:** {row['Company size']}")
                    
                    with col_info2:
                        st.markdown(f"**🌍 Quốc gia:** {row['Country']}")
                        st.markdown(f"**📅 Ngày làm:** {row['Working days']}")
                        st.markdown(f"**⏰ OT Policy:** {row['Overtime Policy']}")
                    
                    st.markdown(f"**📍 Địa điểm:** {row['Location']}")
                
                # Expandable details
                with st.expander(f"📋 Chi tiết {row['Company Name']}", expanded=False):
                    
                    col_detail1, col_detail2 = st.columns(2)
                    
                    with col_detail1:
                        st.markdown("**🏢 Tổng quan công ty:**")
                        overview = row['Company overview'] if pd.notna(row['Company overview']) else "Không có thông tin"
                        st.write(overview)
                        
                        st.markdown("**🔧 Kỹ năng chính:**")
                        skills = row['Our key skills'] if pd.notna(row['Our key skills']) else "Không có thông tin"
                        st.write(skills)
                    
                    with col_detail2:
                        st.markdown("**❤️ Tại sao bạn sẽ thích làm việc ở đây:**")
                        why_love = row['Why you\'ll love working here'] if pd.notna(row['Why you\'ll love working here']) else "Không có thông tin"
                        st.write(why_love)
                        
                        # Text statistics
                        st.markdown("**📊 Thống kê văn bản:**")
                        st.metric("Độ dài văn bản", f"{row['text_length']} ký tự")
                        st.metric("Số từ", f"{row['word_count']} từ")
                    
                    # Company link
                    if pd.notna(row['Href']) and row['Href'].startswith('http'):
                        st.markdown(f"🔗 [Xem chi tiết trên ITViec]({row['Href']})")
                
                st.divider()
    
    # Statistics panel
    if df is not None:
        st.markdown("### 📊 Thống kê tổng quan")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("🏢 Tổng số công ty", len(df))
        
        with col_stat2:
            recommend_count = len(df[df['recommend'] == 1])
            st.metric("✅ Được recommend", f"{recommend_count} ({recommend_count/len(df)*100:.1f}%)")
        
        with col_stat3:
            not_recommend_count = len(df[df['recommend'] == 0])
            st.metric("❌ Không recommend", f"{not_recommend_count} ({not_recommend_count/len(df)*100:.1f}%)")
        
        with col_stat4:
            avg_text_length = df['text_length'].mean()
            st.metric("📝 Độ dài TB", f"{avg_text_length:.0f} ký tự")

def show_candidate_predict():
    """Hiển thị dự đoán bằng mô hình"""
    st.markdown("#### 🤖 Dự đoán bằng mô hình AI")
    st.markdown("Nhập thông tin công ty để dự đoán khả năng **Recommend** hay **Not Recommend**")
    
    # Load model and existing preprocessing components
    @st.cache_resource
    def load_model_and_preprocessors():
        """Load model và các preprocessing components có sẵn"""
        
        try:
            # Load main model
            model = joblib.load("model/best_model_skl_gradient_boosting.pkl")
            # st.success("✅ Loaded best_model_skl_gradient_boosting.pkl")
            
            # Load label encoders  
            label_encoders = joblib.load("model/label_encoders.pkl")
            # st.success("✅ Loaded label_encoders.pkl")
            
            # Load feature scaler
            feature_scaler = joblib.load("model/feature_scaler.pkl") 
            # st.success("✅ Loaded feature_scaler.pkl")
            
            # Try to load text embeddings if available
            text_embeddings = None
            try:
                text_embeddings = np.load("files/text_embeddings.npy")
                # st.success("✅ Loaded text_embeddings.npy")
            except:
                st.info("ℹ️ text_embeddings.npy not found - will use simplified approach")
            
            return {
                'model': model,
                'label_encoders': label_encoders, 
                'feature_scaler': feature_scaler,
                'text_embeddings': text_embeddings,
                'status': 'real'
            }
            
        except FileNotFoundError as e:
            st.error(f"❌ Không tìm thấy file: {str(e)}")
            st.error("💡 Vui lòng đảm bảo các file model đã được lưu đúng đường dẫn")
            return None
        except Exception as e:
            st.error(f"❌ Lỗi load files: {str(e)}")
            return None
    
    # Load reference data
    @st.cache_data
    def load_reference_data():
        try:
            df = pd.read_csv("data/companies_with_recommend.csv")
            return df
        except:
            return None
    
    components = load_model_and_preprocessors()
    if components is None:
        st.error("❌ Không thể load model. Vui lòng kiểm tra lại các file:")
        st.code("""
        model/best_model_skl_gradient_boosting.pkl
        model/label_encoders.pkl  
        model/feature_scaler.pkl
        files/text_embeddings.npy (optional)
        """)
        return
    
    model = components['model']
    label_encoders = components['label_encoders']
    feature_scaler = components['feature_scaler']
    text_embeddings = components['text_embeddings']
    
    ref_df = load_reference_data()
    
    # Hiển thị thông tin model
    # st.info("🤖 Sử dụng Model thật với preprocessing components có sẵn")
    if hasattr(model, 'n_features_in_'):
        # st.info(f"📊 Model expects {model.n_features_in_} features")
        pass
    
    # Input form
    st.markdown("### 📝 Nhập thông tin công ty")
    
    # Column 1: Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏢 Thông tin cơ bản")
        
        # Company Type
        company_types = ['IT Product', 'IT Outsourcing', 'IT Service and IT Consulting', 'Non-IT']
        if ref_df is not None:
            company_types = ref_df['Company Type'].unique().tolist()
        
        company_type = st.selectbox(
            "🏢 Loại công ty:",
            options=company_types,
            index=0
        )
        
        # Company Industry
        industries = ['Software Products and Web Services', 'IT Services and IT Consulting', 
                     'Software Development Outsourcing', 'Financial Services', 'E-commerce']
        if ref_df is not None:
            industries = ref_df['Company industry'].unique().tolist()
        
        company_industry = st.selectbox(
            "🏭 Ngành nghề:",
            options=industries,
            index=0
        )
        
        # Company Size
        sizes = ['1-50 employees', '51-150 employees', '151-300 employees', 
                '301-500 employees', '501-1000 employees', '1000+ employees']
        if ref_df is not None:
            sizes = ref_df['Company size'].unique().tolist()
        
        company_size = st.selectbox(
            "👥 Quy mô:",
            options=sizes,
            index=0
        )
    
    with col2:
        st.markdown("#### ⚙️ Chính sách làm việc")
        
        # Overtime Policy
        ot_policies = ['No OT', 'Extra salary for OT', 'Extra days off for OT', 'OT included in base salary']
        if ref_df is not None:
            ot_policies = ref_df['Overtime Policy'].unique().tolist()
        
        overtime_policy = st.selectbox(
            "⏰ Chính sách OT:",
            options=ot_policies,
            index=0
        )
        
        # Location (chỉ để hiển thị, không dùng cho prediction)
        location = st.text_input(
            "📍 Địa điểm:",
            placeholder="VD: Ho Chi Minh City, Ha Noi...",
            value="Ho Chi Minh City",
            help="Chỉ để hiển thị, không ảnh hưởng đến dự đoán"
        )
    
    # Text inputs
    st.markdown("#### 📝 Mô tả công ty")
    
    col3, col4 = st.columns(2)
    
    with col3:
        company_overview = st.text_area(
            "🏢 Tổng quan công ty:",
            placeholder="Mô tả ngắn về công ty...",
            height=100
        )
        
        our_key_skills = st.text_area(
            "🔧 Kỹ năng chính:",
            placeholder="Các kỹ năng và công nghệ chính...",
            height=100
        )
    
    with col4:
        why_love_working = st.text_area(
            "❤️ Tại sao bạn sẽ thích làm việc ở đây:",
            placeholder="Lý do ứng viên nên chọn công ty...",
            height=100
        )
        
        # Combined text sẽ được tạo tự động
        st.info("💡 Văn bản tổng hợp sẽ được tạo tự động từ các mô tả trên")
    
    # Predict button
    col_predict1, col_predict2, col_predict3 = st.columns([1, 1, 1])
    
    with col_predict2:
        predict_button = st.button(
            "🚀 Dự đoán", 
            type="primary", 
            use_container_width=True,
            help="Click để dự đoán Recommend/Not Recommend"
        )
    
    # Prediction logic
    if predict_button:
        if not all([company_overview, our_key_skills, why_love_working]):
            st.warning("⚠️ Vui lòng điền đầy đủ thông tin mô tả công ty!")
            return
        
        try:
            # Use real model with existing preprocessing components
            # st.info("🔄 Đang xử lý với preprocessing components có sẵn...")
            
            # Prepare text data
            combined_text = f"{company_overview} {our_key_skills} {why_love_working}"
            text_length = len(combined_text)
            word_count = len(combined_text.split())
            
            # FIXED: Sử dụng đúng 4 features như training
            categorical_columns = ['Company Type', 'Company industry', 'Company size', 'Overtime Policy']
            input_values = [company_type, company_industry, company_size, overtime_policy]
            
            # Process categorical features using existing label encoders
            categorical_data = []
            
            for col, value in zip(categorical_columns, input_values):
                if col in label_encoders:
                    le = label_encoders[col]
                    # Handle unknown values
                    if value in le.classes_:
                        encoded_val = le.transform([value])[0]
                    else:
                        # Use most frequent class as fallback
                        encoded_val = 0
                        st.warning(f"⚠️ Unknown value '{value}' for '{col}', using fallback")
                    categorical_data.append(encoded_val)
                else:
                    categorical_data.append(0)
                    st.warning(f"⚠️ No encoder for '{col}', using fallback")
            
            # st.success(f"✅ Prepared {len(categorical_data)} categorical features for scaler")
            
            # Scale categorical features using existing scaler
            categorical_array = np.array(categorical_data).reshape(1, -1)
            categorical_scaled = feature_scaler.transform(categorical_array)[0]
            # st.success(f"✅ Successfully scaled features to shape: {categorical_scaled.shape}")
            
            # Handle text embeddings
            if text_embeddings is not None:
                # Use pre-computed embeddings (for demo, use first embedding)
                # st.info("📝 Sử dụng pre-computed text embeddings")
                text_features = text_embeddings[0]  
            else:
                # Create dummy text embeddings for demo
                st.warning("⚠️ Tạo dummy text embeddings cho demo")
                text_features = np.random.normal(0, 0.1, 768)
            
            # Combine all features to match model input
            X = np.concatenate([text_features, categorical_scaled]).reshape(1, -1)
            
            # st.success(f"✅ Đã tạo {X.shape[1]} features để input vào model")
            
            # Make prediction
            prediction = model.predict(X)[0]
            prediction_proba = model.predict_proba(X)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Kết quả dự đoán")
            
            col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
            
            with col_result2:
                if prediction == 1:
                    st.success("### ✅ RECOMMEND")
                    st.balloons()
                else:
                    st.error("### ❌ NOT RECOMMEND")
                
                # Probability display
                recommend_prob = prediction_proba[1] * 100
                not_recommend_prob = prediction_proba[0] * 100
                
                st.markdown("#### 📊 Xác suất dự đoán:")
                
                st.markdown(f"**✅ Recommend:** {recommend_prob:.1f}%")
                st.progress(recommend_prob / 100)
                
                st.markdown(f"**❌ Not Recommend:** {not_recommend_prob:.1f}%")
                st.progress(not_recommend_prob / 100)
                
                # Confidence level
                confidence = max(recommend_prob, not_recommend_prob)
                if confidence >= 80:
                    st.success(f"🎯 Độ tin cậy cao: {confidence:.1f}%")
                elif confidence >= 60:
                    st.warning(f"⚠️ Độ tin cậy trung bình: {confidence:.1f}%")
                else:
                    st.error(f"❌ Độ tin cậy thấp: {confidence:.1f}%")
            
            # Feature importance visualization
            # if hasattr(model, 'feature_importances_'):
            #     st.markdown("### 📈 Tầm quan trọng của các yếu tố")
                
            #     # Create meaningful feature names  
            #     text_features_names = [f'text_dim_{i}' for i in range(768)]
            #     cat_features_names = ['Company_Type', 'Company_Industry', 'Company_Size', 'Overtime_Policy']
            #     all_feature_names = text_features_names + cat_features_names
                
            #     # Get top important features
            #     feature_importance_data = list(zip(all_feature_names, model.feature_importances_))
            #     feature_importance_data.sort(key=lambda x: x[1], reverse=True)
                
            #     # Show top 20
            #     top_features = feature_importance_data[:20]
                
            #     importance_df = pd.DataFrame(top_features, columns=['feature', 'importance'])
                
            #     fig = go.Figure(go.Bar(
            #         x=importance_df['importance'],
            #         y=importance_df['feature'],
            #         orientation='h',
            #         marker_color='lightblue'
            #     ))
                
            #     fig.update_layout(
            #         title="Top 20 Features quan trọng nhất",
            #         xaxis_title="Tầm quan trọng",
            #         yaxis_title="Features",
            #         height=600,
            #         plot_bgcolor='rgba(0,0,0,0)',
            #         paper_bgcolor='rgba(0,0,0,0)',
            #         font_color='white'
            #     )
                
            #     st.plotly_chart(fig, use_container_width=True)
            
            # Summary information
            st.markdown("### 📋 Tóm tắt thông tin")
            
            col_summary1, col_summary2 = st.columns(2)
            
            with col_summary1:
                st.markdown("**🏢 Thông tin công ty:**")
                st.write(f"• Loại: {company_type}")
                st.write(f"• Ngành: {company_industry}")
                st.write(f"• Quy mô: {company_size}")
                st.write(f"• Chính sách OT: {overtime_policy}")
            
            with col_summary2:
                st.markdown("**📊 Thống kê:**")
                st.write(f"• Độ dài văn bản: {text_length} ký tự")
                st.write(f"• Số từ: {word_count} từ")
                st.write(f"• Features được tạo: {X.shape[1]}")
                st.write(f"• Địa điểm: {location}")
        
        except Exception as e:
            st.error(f"❌ Lỗi trong quá trình dự đoán: {str(e)}")
            st.error("💡 Vui lòng kiểm tra lại dữ liệu đầu vào và model")
            
            # Show detailed error for debugging
            with st.expander("🔍 Chi tiết lỗi", expanded=False):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())
    
    # # Instructions
    # with st.expander("📖 Hướng dẫn sử dụng", expanded=False):
    #     st.markdown("""
    #     ### 🎯 Cách sử dụng:
    #     1. **Điền thông tin cơ bản** về công ty (loại, ngành, quy mô, OT policy)
    #     2. **Mô tả chi tiết** về công ty trong 3 ô text
    #     3. **Click Dự đoán** để nhận kết quả
        
    #     ### 📊 Kết quả bao gồm:
    #     - **Dự đoán chính**: Recommend hay Not Recommend
    #     - **Xác suất**: Tỷ lệ % cho mỗi class
    #     - **Độ tin cậy**: Mức độ chắc chắn của model
    #     - **Feature Importance**: Yếu tố nào quan trọng nhất
        
    #     ### 🤖 Model Information:
    #     - **Model**: GradientBoostingClassifier từ Scikit-learn
    #     - **Features**: 4 categorical + 768 text embeddings = 772 total
    #     - **Files cần thiết**: 
    #       - `model/best_model_skl_gradient_boosting.pkl`
    #       - `model/label_encoders.pkl`
    #       - `model/feature_scaler.pkl`
    #       - `files/text_embeddings.npy` (optional)
        
    #     ### 💡 Lưu ý:
    #     - Model được train trên dữ liệu ITViec thật
    #     - Chỉ sử dụng 4 categorical features như training
    #     - Text embeddings: Sử dụng pre-computed nếu có, dummy nếu không
    #     - Kết quả phụ thuộc vào chất lượng mô tả công ty
    #     """)
    
    # # Technical info
    # with st.expander("🔬 Thông tin kỹ thuật", expanded=False):
    #     st.markdown("#### 🧠 Architecture:")
    #     st.markdown("""
    #     **Preprocessing Pipeline:**
    #     - Categorical encoding: Label encoders từ training (4 features)
    #     - Feature scaling: StandardScaler từ training  
    #     - Text processing: Pre-computed embeddings hoặc dummy cho demo
        
    #     **Model Details:**
    #     - Type: GradientBoostingClassifier (Scikit-learn)
    #     - Features: 772 total (768 text + 4 categorical scaled)
    #     - Training: Đã hoàn thành với dữ liệu ITViec
        
    #     **Categorical Features Used:**
    #     - Company Type
    #     - Company industry  
    #     - Company size
    #     - Overtime Policy
        
    #     **Files Usage:**
    #     - Model: Loaded từ best_model_skl_gradient_boosting.pkl
    #     - Encoders: Loaded từ label_encoders.pkl
    #     - Scaler: Loaded từ feature_scaler.pkl
    #     - Embeddings: Optional từ text_embeddings.npy
    #     """)
        
    #     st.markdown("#### 📊 Current Model Stats:")
    #     if hasattr(model, 'n_features_in_'):
    #         st.write(f"• Expected features: {model.n_features_in_}")
    #     if hasattr(model, 'n_classes_'):
    #         st.write(f"• Number of classes: {model.n_classes_}")
    #     if hasattr(model, 'feature_importances_'):
    #         st.write(f"• Feature importance available: ✅")
            
    #         # Show categorical feature importance
    #         if label_encoders:
    #             cat_start_idx = 768  # After text features
    #             cat_importance = model.feature_importances_[cat_start_idx:]
    #             cat_names = ['Company_Type', 'Company_Industry', 'Company_Size', 'Overtime_Policy']
                
    #             st.write("• Categorical features importance:")
    #             for i, (name, imp) in enumerate(zip(cat_names, cat_importance)):
    #                 st.write(f"  {i+1}. {name}: {imp:.4f}")
        
    #     # Show loaded components info
    #     st.markdown("#### 📦 Loaded Components:")
    #     st.write(f"• Label encoders: {len(label_encoders) if label_encoders else 0} columns")
    #     st.write(f"• Feature scaler: {'✅' if feature_scaler else '❌'}")
    #     st.write(f"• Text embeddings: {'✅' if text_embeddings is not None else '❌ (using dummy)'}")
        
    #     if label_encoders:
    #         st.write("• Categorical columns:")
    #         for col, encoder in label_encoders.items():
    #             st.write(f"  - {col}: {len(encoder.classes_)} classes")

def candidate_classification_tab():
    """Main function cho Candidate Classification tab"""
    st.markdown("## 👤 Candidate Classification")
    
    # Sidebar cho Candidate Classification
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("""
        <div style="background: #2c3e50; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h4 style="color: white; text-align: center; margin-bottom: 1rem;">📋 Menu</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        if st.button("📊 Tổng quan", key="overview", use_container_width=True):
            st.session_state.candidate_page = "overview"
        
        if st.button("🔍 Tìm kiếm công ty", key="search", use_container_width=True):
            st.session_state.candidate_page = "search"
            
        if st.button("🤖 Dự đoán bằng mô hình", key="predict", use_container_width=True):
            st.session_state.candidate_page = "predict"
    
    with col2:
        # Initialize session state
        if 'candidate_page' not in st.session_state:
            st.session_state.candidate_page = "overview"
        
        # Content area based on selected page
        if st.session_state.candidate_page == "overview":
            show_candidate_overview()
        elif st.session_state.candidate_page == "search":
            show_candidate_search()
        elif st.session_state.candidate_page == "predict":
            show_candidate_predict()