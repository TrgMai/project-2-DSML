import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def preprocess_company_data(self, df):
        """Preprocess company data for similarity calculation"""
        processed_df = df.copy()
        
        # Clean text data
        if 'company_name' in processed_df.columns:
            processed_df['company_name_clean'] = processed_df['company_name'].str.lower().str.strip()
        
        # Encode categorical variables
        categorical_cols = ['industry', 'company_size', 'location']
        for col in categorical_cols:
            if col in processed_df.columns:
                le = LabelEncoder()
                processed_df[f'{col}_encoded'] = le.fit_transform(processed_df[col].astype(str))
                self.label_encoders[col] = le
        
        # Normalize numerical features
        numerical_cols = ['founded_year', 'employee_count']
        for col in numerical_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
        return processed_df
    
    def preprocess_candidate_data(self, candidate_info):
        """Preprocess candidate data for classification"""
        features = {}
        
        # Age normalization
        features['age_normalized'] = (candidate_info.get('age', 25) - 25) / 20
        
        # Experience normalization
        features['experience_normalized'] = candidate_info.get('experience', 0) / 10
        
        # Education encoding
        education_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
        features['education_encoded'] = education_mapping.get(candidate_info.get('education', 'Bachelor'), 2)
        
        # Skills count
        skills = candidate_info.get('skills', [])
        features['skills_count'] = len(skills) if isinstance(skills, list) else 0
        
        # Salary normalization
        features['salary_normalized'] = candidate_info.get('salary_expectation', 25) / 100
        
        # Location encoding (simplified)
        location_mapping = {'Ho Chi Minh City': 1, 'Hanoi': 2, 'Da Nang': 3, 'Other': 0}
        features['location_encoded'] = location_mapping.get(candidate_info.get('location', 'Other'), 0)
        
        return np.array(list(features.values())).reshape(1, -1)
    
    def calculate_company_similarity(self, company1_features, company2_features):
        """Calculate similarity between two companies"""
        # Simple cosine similarity for demo
        dot_product = np.dot(company1_features, company2_features)
        norm1 = np.linalg.norm(company1_features)
        norm2 = np.linalg.norm(company2_features)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0, similarity)  # Ensure non-negative
    
    def extract_text_features(self, text):
        """Extract features from text data"""
        if not isinstance(text, str):
            return []
        
        # Simple text processing
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return words
    
    def create_feature_vector(self, data_dict):
        """Create feature vector from dictionary"""
        features = []
        
        for key, value in data_dict.items():
            if isinstance(value, (int, float)):
                features.append(value)
            elif isinstance(value, str):
                # Simple encoding for strings
                features.append(hash(value) % 1000 / 1000)
            elif isinstance(value, list):
                features.append(len(value))
        
        return np.array(features)