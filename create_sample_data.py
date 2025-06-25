# Script to generate sample CSV files
# Run this once to create sample data

import pandas as pd
import numpy as np
import os

def create_sample_data():
    """Create sample CSV files for the project"""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # 1. Companies data
    np.random.seed(42)
    companies_data = {
        'company_id': range(1, 101),
        'company_name': [
            'TechCorp Vietnam', 'DataSoft Solutions', 'AI Ventures Ltd', 'CloudMaster Pro', 'DevHub Studio',
            'FinTech Innovations', 'CyberGuard Security', 'WebSolutions Inc', 'MobileTech Labs', 'GameStudio Pro',
            'EcommerceGiant VN', 'SocialConnect App', 'HealthTech Medical', 'EduPlatform Online', 'GreenEnergy Corp',
            'SmartCity Solutions', 'BlockChain Technologies', 'IoT Systems Vietnam', 'VR Innovations Lab', 'RoboTech Manufacturing',
            'FoodDelivery Express', 'TravelApp Vietnam', 'MusicStream Platform', 'PhotoShare Social', 'VideoEdit Pro',
            'ChatBot AI Solutions', 'TranslateNow Service', 'WeatherApp Vietnam', 'NewsPortal Digital', 'SportsTech Analytics',
            'FashionTech Retail', 'BeautyApp Vietnam', 'PetCare Services', 'HomeTech Automation', 'CarTech Solutions',
            'BikeShare Vietnam', 'Delivery Express Co', 'Logistics Pro Vietnam', 'Supply Chain Solutions', 'Inventory Management',
            'HR Solutions Vietnam', 'Payroll Tech Service', 'Accounting App Pro', 'LegalTech Vietnam', 'InsureTech Solutions',
            'RealEstate Platform', 'ConstructTech Vietnam', 'AgriTech Solutions', 'Mining Data Analytics', 'Ocean Tech Marine'
        ] + [f'Company {i}' for i in range(51, 101)],
        'industry': np.random.choice([
            'Information Technology', 'Financial Services', 'Healthcare', 'Education', 'E-commerce', 
            'Manufacturing', 'Retail', 'Logistics', 'Real Estate', 'Agriculture'
        ], 100),
        'company_size': np.random.choice(['Startup (1-50)', 'Small (51-200)', 'Medium (201-1000)', 'Large (1000+)'], 100),
        'location': np.random.choice(['Ho Chi Minh City', 'Hanoi', 'Da Nang', 'Can Tho', 'Hai Phong'], 100),
        'founded_year': np.random.randint(1995, 2024, 100),
        'employee_count': np.random.randint(10, 5000, 100),
        'revenue_million_usd': np.random.randint(1, 500, 100),
        'description': [
            f'Leading technology company specializing in innovative solutions for industry {i%5+1}'
            for i in range(100)
        ],
        'website': [f'https://company{i}.com' for i in range(1, 101)],
        'benefits': np.random.choice([
            'Health Insurance, Flexible Hours, Remote Work',
            'Competitive Salary, Training, Career Development',
            'Performance Bonus, Team Building, Modern Office',
            'Stock Options, Free Lunch, Gym Membership'
        ], 100)
    }
    
    companies_df = pd.DataFrame(companies_data)
    companies_df.to_csv('data/companies_data.csv', index=False, encoding='utf-8')
    
    # 2. Candidates data
    first_names = ['Nguyen', 'Tran', 'Le', 'Pham', 'Hoang', 'Huynh', 'Vo', 'Dang', 'Bui', 'Do']
    last_names = ['Van A', 'Thi B', 'Minh C', 'Hoang D', 'Thanh E', 'Quang F', 'Thu G', 'Duc H', 'Mai I', 'Linh J']
    
    candidates_data = {
        'candidate_id': range(1, 501),
        'full_name': [f'{np.random.choice(first_names)} {np.random.choice(last_names)} {i}' for i in range(1, 501)],
        'age': np.random.randint(22, 50, 500),
        'gender': np.random.choice(['Male', 'Female', 'Other'], 500),
        'experience_years': np.random.randint(0, 20, 500),
        'education_level': np.random.choice(['High School', 'Associate', 'Bachelor', 'Master', 'PhD'], 500),
        'major': np.random.choice([
            'Computer Science', 'Information Technology', 'Software Engineering', 'Data Science',
            'Business Administration', 'Marketing', 'Finance', 'Electrical Engineering',
            'Mechanical Engineering', 'English Literature', 'Economics', 'Mathematics'
        ], 500),
        'skills': [
            '|'.join(np.random.choice([
                'Python', 'Java', 'JavaScript', 'C++', 'SQL', 'HTML/CSS', 'React', 'Angular',
                'Node.js', 'Django', 'Flask', 'Spring', 'Machine Learning', 'Data Analysis',
                'Project Management', 'Team Leadership', 'Communication', 'Problem Solving',
                'Git', 'Docker', 'AWS', 'Azure', 'Agile', 'Scrum'
            ], size=np.random.randint(3, 8), replace=False))
            for _ in range(500)
        ],
        'salary_expectation_usd': np.random.randint(800, 5000, 500),
        'preferred_location': np.random.choice(['Ho Chi Minh City', 'Hanoi', 'Da Nang', 'Remote', 'Flexible'], 500),
        'job_type_preference': np.random.choice(['Full-time', 'Part-time', 'Contract', 'Internship'], 500),
        'current_employment_status': np.random.choice(['Employed', 'Unemployed', 'Student', 'Freelancer'], 500),
        'english_level': np.random.choice(['Basic', 'Intermediate', 'Advanced', 'Native'], 500),
        'availability': np.random.choice(['Immediate', '2 weeks', '1 month', '2 months'], 500),
        'portfolio_url': [f'https://portfolio{i}.com' if np.random.random() > 0.3 else '' for i in range(1, 501)],
        'linkedin_url': [f'https://linkedin.com/in/candidate{i}' for i in range(1, 501)]
    }
    
    candidates_df = pd.DataFrame(candidates_data)
    candidates_df.to_csv('data/candidates_data.csv', index=False, encoding='utf-8')
    
    # 3. Sample predictions data (for visualization)
    predictions_data = {
        'prediction_id': range(1, 201),
        'candidate_id': np.random.choice(range(1, 501), 200),
        'company_id': np.random.choice(range(1, 101), 200),
        'recommendation_score': np.random.uniform(0.1, 1.0, 200),
        'predicted_class': np.random.choice(['Recommend', 'Not Recommend'], 200),
        'confidence': np.random.uniform(0.6, 0.99, 200),
        'prediction_date': pd.date_range('2024-01-01', periods=200, freq='D'),
        'model_version': np.random.choice(['v1.0', 'v1.1', 'v1.2'], 200),
        'features_used': [
            '|'.join(['skills_match', 'experience_fit', 'education_level', 'salary_fit', 'location_match'])
            for _ in range(200)
        ]
    }
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv('data/sample_predictions.csv', index=False, encoding='utf-8')
    
    # 4. Job positions data (for matching)
    positions_data = {
        'position_id': range(1, 51),
        'company_id': np.random.choice(range(1, 101), 50),
        'job_title': np.random.choice([
            'Software Engineer', 'Data Scientist', 'Product Manager', 'UI/UX Designer',
            'DevOps Engineer', 'Business Analyst', 'QA Engineer', 'Marketing Specialist',
            'Sales Manager', 'HR Specialist', 'Financial Analyst', 'Project Manager',
            'Full Stack Developer', 'Mobile Developer', 'Backend Developer', 'Frontend Developer'
        ], 50),
        'required_skills': [
            '|'.join(np.random.choice([
                'Python', 'Java', 'JavaScript', 'SQL', 'React', 'Angular', 'Node.js',
                'Machine Learning', 'Data Analysis', 'Project Management', 'Communication',
                'Problem Solving', 'Leadership', 'Agile', 'Scrum'
            ], size=np.random.randint(3, 6), replace=False))
            for _ in range(50)
        ],
        'min_experience': np.random.randint(0, 5, 50),
        'max_experience': np.random.randint(5, 15, 50),
        'education_requirement': np.random.choice(['Bachelor', 'Master', 'Any'], 50),
        'salary_range_min': np.random.randint(500, 2000, 50),
        'salary_range_max': np.random.randint(2000, 8000, 50),
        'job_type': np.random.choice(['Full-time', 'Part-time', 'Contract'], 50),
        'remote_allowed': np.random.choice([True, False], 50),
        'posted_date': pd.date_range('2024-01-01', periods=50, freq='W'),
        'application_deadline': pd.date_range('2024-03-01', periods=50, freq='W')
    }
    
    positions_df = pd.DataFrame(positions_data)
    positions_df.to_csv('data/job_positions.csv', index=False, encoding='utf-8')
    
    print("Sample data files created successfully!")
    print("Files created:")
    print("- data/companies_data.csv (100 companies)")
    print("- data/candidates_data.csv (500 candidates)")  
    print("- data/sample_predictions.csv (200 predictions)")
    print("- data/job_positions.csv (50 job positions)")

if __name__ == "__main__":
    create_sample_data()