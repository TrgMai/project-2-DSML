import pickle
import streamlit as st
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    models = {}
    
    try:
        # Try to load real models
        if os.path.exists('models/company_similarity_model.pkl'):
            with open('models/company_similarity_model.pkl', 'rb') as f:
                models['company_similarity'] = pickle.load(f)
        else:
            # Create fake model for demo
            models['company_similarity'] = create_fake_similarity_model()
            
        if os.path.exists('models/candidate_recommendation_model.pkl'):
            with open('models/candidate_recommendation_model.pkl', 'rb') as f:
                models['candidate_classification'] = pickle.load(f)
        else:
            # Create fake model for demo
            models['candidate_classification'] = create_fake_classification_model()
            
    except Exception as e:
        st.warning(f"Could not load models: {str(e)}. Using demo models.")
        models['company_similarity'] = create_fake_similarity_model()
        models['candidate_classification'] = create_fake_classification_model()
    
    return models

def create_fake_similarity_model():
    """Create a fake similarity model for demo"""
    # This is just for demo - replace with your actual model
    return TfidfVectorizer(max_features=1000)

def create_fake_classification_model():
    """Create a fake classification model for demo"""
    # This is just for demo - replace with your actual model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fake training data
    X_fake = np.random.rand(100, 10)
    y_fake = np.random.choice([0, 1], 100)
    model.fit(X_fake, y_fake)
    return model

def save_model(model, model_name):
    """Save model to pickle file"""
    try:
        os.makedirs('models', exist_ok=True)
        with open(f'models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False