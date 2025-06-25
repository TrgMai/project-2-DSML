import pandas as pd
import numpy as np
import streamlit as st
import os

@st.cache_data
def load_sample_data(data_type):
    """Load sample data for demo purposes"""
    try:
        if data_type == 'companies':
            return load_companies_data()
        elif data_type == 'candidates':
            return load_candidates_data()
        else:
            return None
    except Exception as e:
        st.error(f"Error loading {data_type} data: {str(e)}")
        return None

def load_companies_data():
    """Generate fake companies data"""
    companies_data = {
        'company_id': range(1, 51),
        'company_name': [
            'TechCorp', 'DataSoft', 'AIVentures', 'CloudMaster', 'DevHub',
            'FinTech Pro', 'CyberGuard', 'WebSolutions', 'MobileTech', 'GameStudio',
            'EcommerceGiant', 'SocialConnect', 'HealthTech', 'EduPlatform', 'GreenEnergy',
            'SmartCity', 'BlockChain Co', 'IoT Systems', 'VR Innovations', 'RoboTech',
            'FoodDelivery', 'TravelApp', 'MusicStream', 'PhotoShare', 'VideoEdit',
            'ChatBot AI', 'TranslateNow', 'WeatherApp', 'NewsPortal', 'SportsTech',
            'FashionTech', 'BeautyApp', 'PetCare', 'HomeTech', 'CarTech',
            'BikeShare', 'Delivery Co', 'Logistics Pro', 'Supply Chain', 'Inventory',
            'HR Solutions', 'Payroll Tech', 'Accounting App', 'Legal Tech', 'InsureTech',
            'RealEstate', 'ConstructTech', 'AgriTech', 'Mining Data', 'Ocean Tech'
        ],
        'industry': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Education', 'E-commerce'], 50),
        'company_size': np.random.choice(['Startup', 'Small', 'Medium', 'Large'], 50),
        'location': np.random.choice(['Ho Chi Minh City', 'Hanoi', 'Da Nang', 'Can Tho'], 50),
        'founded_year': np.random.randint(2000, 2024, 50),
        'employee_count': np.random.randint(10, 5000, 50)
    }
    return pd.DataFrame(companies_data)

def load_candidates_data():
    """Generate fake candidates data"""
    candidates_data = {
        'candidate_id': range(1, 101),
        'name': [f'Candidate {i}' for i in range(1, 101)],
        'age': np.random.randint(22, 45, 100),
        'experience_years': np.random.randint(0, 15, 100),
        'education': np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], 100),
        'skills': ['Python,SQL,ML' for _ in range(100)],  # Simplified for demo
        'salary_expectation': np.random.randint(15, 80, 100),
        'location': np.random.choice(['Ho Chi Minh City', 'Hanoi', 'Da Nang'], 100),
        'job_type_pref': np.random.choice(['Full-time', 'Part-time', 'Contract'], 100)
    }
    return pd.DataFrame(candidates_data)