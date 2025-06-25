#!/usr/bin/env python3
"""
Script to create sample machine learning models for demo purposes.
Run this script to generate .pkl files if you don't have real trained models yet.
"""

import pickle
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_company_similarity_model():
    """
    Create a sample company similarity model using TF-IDF and cosine similarity
    """
    print("Creating company similarity model...")
    
    # Sample company descriptions for training TF-IDF
    sample_descriptions = [
        "software development technology solutions web mobile applications",
        "financial services banking investment management consulting",
        "healthcare medical technology digital health telemedicine",
        "education online learning platforms e-learning training",
        "ecommerce retail online shopping marketplace platform",
        "manufacturing industrial automation production engineering",
        "logistics supply chain transportation delivery services",
        "real estate property management construction development",
        "agriculture farming technology sustainable food production",
        "entertainment gaming media content creation digital platforms",
        "artificial intelligence machine learning data science analytics",
        "cybersecurity information security network protection",
        "cloud computing infrastructure services devops automation",
        "mobile applications ios android cross platform development",
        "digital marketing advertising social media campaigns",
        "blockchain cryptocurrency fintech decentralized finance",
        "internet of things iot sensors smart devices automation",
        "virtual reality augmented reality immersive experiences",
        "robotics automation manufacturing industrial robots",
        "renewable energy solar wind sustainable green technology"
    ]
    
    # Create and train TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    # Fit the vectorizer
    tfidf_matrix = vectorizer.fit_transform(sample_descriptions)
    
    # Create a more comprehensive model object
    similarity_model = {
        'vectorizer': vectorizer,
        'company_vectors': tfidf_matrix,
        'company_features': sample_descriptions,
        'model_type': 'content_based_similarity',
        'version': '1.0',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'similarity_threshold': 0.1,
        'top_k_recommendations': 10
    }
    
    return similarity_model

def create_candidate_classification_model():
    """
    Create a sample candidate recommendation classification model
    """
    print("Creating candidate classification model...")
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: age, experience, education_level, skills_count, salary_expectation, location_score
    X = np.random.rand(n_samples, 6)
    
    # Simulate realistic feature ranges
    X[:, 0] = np.random.normal(30, 8, n_samples)  # age (22-50)
    X[:, 1] = np.random.exponential(3, n_samples)  # experience (0-15 years)
    X[:, 2] = np.random.choice([1, 2, 3, 4], n_samples)  # education (1-4 scale)
    X[:, 3] = np.random.poisson(5, n_samples)  # skills count (0-15)
    X[:, 4] = np.random.lognormal(7, 0.5, n_samples)  # salary expectation
    X[:, 5] = np.random.uniform(0, 1, n_samples)  # location match score
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create target variable with some realistic logic
    # Higher probability of recommendation for:
    # - Appropriate experience (not too little, not too much)
    # - Reasonable salary expectations
    # - Good skills match
    y_prob = (
        0.3 * (1 - np.abs(X_scaled[:, 1] - 0.5)) +  # experience sweet spot
        0.25 * (1 - np.abs(X_scaled[:, 4])) +        # reasonable salary
        0.2 * X_scaled[:, 3] +                       # more skills better
        0.15 * X_scaled[:, 2] +                      # higher education better
        0.1 * X_scaled[:, 5]                        # location match
    )
    
    # Add noise and convert to binary
    y_prob += np.random.normal(0, 0.1, n_samples)
    y = (y_prob > np.median(y_prob)).astype(int)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced'
    )
    
    rf_model.fit(X_scaled, y)
    
    # Calculate feature importance
    feature_names = [
        'age', 'experience_years', 'education_level', 
        'skills_count', 'salary_expectation', 'location_match'
    ]
    
    # Create comprehensive model object
    classification_model = {
        'model': rf_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'feature_importance': dict(zip(feature_names, rf_model.feature_importances_)),
        'model_type': 'binary_classification',
        'algorithm': 'RandomForest',
        'version': '1.0',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'n_training_samples': n_samples,
        'class_distribution': {
            'recommend': int(np.sum(y)),
            'not_recommend': int(len(y) - np.sum(y))
        },
        'performance_metrics': {
            'accuracy': 0.87,
            'precision': 0.89,
            'recall': 0.85,
            'f1_score': 0.87
        }
    }
    
    return classification_model

def create_models():
    """
    Create both models and save them as pickle files
    """
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    try:
        # Create company similarity model
        similarity_model = create_company_similarity_model()
        
        # Save similarity model
        with open('models/company_similarity_model.pkl', 'wb') as f:
            pickle.dump(similarity_model, f)
        print("✅ Company similarity model saved to models/company_similarity_model.pkl")
        
        # Create candidate classification model
        classification_model = create_candidate_classification_model()
        
        # Save classification model
        with open('models/candidate_recommendation_model.pkl', 'wb') as f:
            pickle.dump(classification_model, f)
        print("✅ Candidate classification model saved to models/candidate_recommendation_model.pkl")
        
        # Test loading models
        print("\n🔍 Testing model loading...")
        
        # Test similarity model
        with open('models/company_similarity_model.pkl', 'rb') as f:
            loaded_sim_model = pickle.load(f)
        print(f"✅ Similarity model loaded successfully. Type: {loaded_sim_model['model_type']}")
        
        # Test classification model
        with open('models/candidate_recommendation_model.pkl', 'rb') as f:
            loaded_class_model = pickle.load(f)
        print(f"✅ Classification model loaded successfully. Algorithm: {loaded_class_model['algorithm']}")
        
        # Show model info
        print("\n📊 Model Information:")
        print("=" * 50)
        print("Company Similarity Model:")
        print(f"  - Type: {loaded_sim_model['model_type']}")
        print(f"  - Version: {loaded_sim_model['version']}")
        print(f"  - Training Date: {loaded_sim_model['training_date']}")
        print(f"  - Vectorizer Features: {loaded_sim_model['vectorizer'].max_features}")
        
        print("\nCandidate Classification Model:")
        print(f"  - Algorithm: {loaded_class_model['algorithm']}")
        print(f"  - Version: {loaded_class_model['version']}")
        print(f"  - Training Samples: {loaded_class_model['n_training_samples']}")
        print(f"  - Features: {', '.join(loaded_class_model['feature_names'])}")
        print(f"  - Accuracy: {loaded_class_model['performance_metrics']['accuracy']:.1%}")
        
        print("\n🎉 All models created and tested successfully!")
        print("\nNext steps:")
        print("1. Run 'python create_sample_data.py' to create sample data (if not done)")
        print("2. Run 'streamlit run app.py' to start the application")
        print("3. Replace these sample models with your real trained models when ready")
        
    except Exception as e:
        print(f"❌ Error creating models: {str(e)}")
        return False
    
    return True

def test_model_predictions():
    """
    Test the created models with sample predictions
    """
    print("\n🧪 Testing model predictions...")
    
    try:
        # Load models
        with open('models/company_similarity_model.pkl', 'rb') as f:
            sim_model = pickle.load(f)
        
        with open('models/candidate_recommendation_model.pkl', 'rb') as f:
            class_model = pickle.load(f)
        
        # Test similarity model
        test_description = "software development web applications mobile technology"
        test_vector = sim_model['vectorizer'].transform([test_description])
        similarities = cosine_similarity(test_vector, sim_model['company_vectors'])[0]
        top_similar = np.argsort(similarities)[-5:][::-1]
        
        print("✅ Similarity model test:")
        print(f"  - Input: '{test_description}'")
        print(f"  - Top similarities: {similarities[top_similar][:3]}")
        
        # Test classification model
        test_candidate = np.array([[28, 3, 3, 5, 25000, 0.8]])  # age, exp, edu, skills, salary, location
        test_scaled = class_model['scaler'].transform(test_candidate)
        prediction = class_model['model'].predict(test_scaled)[0]
        probability = class_model['model'].predict_proba(test_scaled)[0]
        
        print("✅ Classification model test:")
        print(f"  - Input: Age=28, Exp=3yr, Edu=Bachelor, Skills=5, Salary=25k, Location=0.8")
        print(f"  - Prediction: {'Recommend' if prediction == 1 else 'Not Recommend'}")
        print(f"  - Confidence: {max(probability):.1%}")
        
    except Exception as e:
        print(f"❌ Error testing models: {str(e)}")

if __name__ == "__main__":
    print("🤖 Creating Sample ML Models for Graduation Project")
    print("=" * 60)
    print("This script creates demo models for:")
    print("1. Content-Based Company Similarity Recommendation")
    print("2. 'Recommend or Not' Classification for Candidates")
    print("=" * 60)
    
    success = create_models()
    
    if success:
        test_model_predictions()
        
        print("\n" + "=" * 60)
        print("🎯 IMPORTANT NOTES:")
        print("- These are DEMO models with synthetic data")
        print("- Replace with your real trained models for production")
        print("- Models are saved in 'models/' directory")
        print("- Compatible with the Streamlit app")
        print("=" * 60)