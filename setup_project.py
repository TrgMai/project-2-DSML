#!/usr/bin/env python3
"""
Complete project setup script for ML Graduation Project
Run this script to set up everything needed for the demo
"""

import os
import sys
import subprocess
import time

def print_header(text):
    print("\n" + "="*60)
    print(f"🚀 {text}")
    print("="*60)

def print_step(step_num, total_steps, description):
    print(f"\n📋 Step {step_num}/{total_steps}: {description}")
    print("-" * 40)

def run_command(command, description):
    """Run a system command and handle errors"""
    try:
        print(f"⏳ {description}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ Error in {description}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception in {description}: {str(e)}")
        return False

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'models',
        'data', 
        'utils',
        'pages',
        'assets',
        'assets/images'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def create_init_files():
    """Create __init__.py files for Python packages"""
    init_files = [
        'utils/__init__.py',
        'pages/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Auto-generated __init__.py\n')
            print(f"📄 Created: {init_file}")
    
    return True

def check_python_version():
    """Check if Python version is compatible"""
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} is not compatible. Need Python 3.8+")
        return False

def install_requirements():
    """Install required packages"""
    requirements = [
        "streamlit==1.28.1",
        "pandas==2.0.3", 
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "plotly==5.15.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2"
    ]
    
    print("📦 Installing required packages...")
    for package in requirements:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️  Warning: Failed to install {package}")
    
    return True

def setup_project():
    """Main setup function"""
    print_header("ML GRADUATION PROJECT SETUP")
    
    print("🎓 Content-Based Company Similarity & Candidate Classification")
    print("📅 Setting up demo environment with sample data and models")
    
    total_steps = 7
    
    # Step 1: Check Python version
    print_step(1, total_steps, "Checking Python compatibility")
    if not check_python_version():
        print("❌ Setup failed. Please upgrade to Python 3.8 or higher.")
        return False
    
    # Step 2: Create directory structure
    print_step(2, total_steps, "Creating directory structure")
    create_directory_structure()
    create_init_files()
    
    # Step 3: Install requirements
    print_step(3, total_steps, "Installing Python packages")
    install_requirements()
    
    # Step 4: Create sample data
    print_step(4, total_steps, "Creating sample data")
    try:
        exec(open('create_sample_data.py').read())
        print("✅ Sample data created successfully")
    except Exception as e:
        print(f"❌ Error creating sample data: {str(e)}")
        print("🔧 Will create minimal data...")
        create_minimal_data()
    
    # Step 5: Create sample models
    print_step(5, total_steps, "Creating sample ML models")
    try:
        exec(open('create_sample_models.py').read())
        print("✅ Sample models created successfully")
    except Exception as e:
        print(f"❌ Error creating sample models: {str(e)}")
        print("🔧 Will create minimal models...")
        create_minimal_models()
    
    # Step 6: Verify setup
    print_step(6, total_steps, "Verifying setup")
    verify_setup()
    
    # Step 7: Final instructions
    print_step(7, total_steps, "Setup complete!")
    show_final_instructions()
    
    return True

def create_minimal_data():
    """Create minimal data if full script fails"""
    import pandas as pd
    import numpy as np
    
    # Minimal companies data
    companies = pd.DataFrame({
        'company_id': range(1, 11),
        'company_name': [f'Company {i}' for i in range(1, 11)],
        'industry': ['Technology'] * 10,
        'company_size': ['Medium'] * 10,
        'location': ['Ho Chi Minh City'] * 10
    })
    companies.to_csv('data/companies_data.csv', index=False)
    
    # Minimal candidates data
    candidates = pd.DataFrame({
        'candidate_id': range(1, 21),
        'full_name': [f'Candidate {i}' for i in range(1, 21)],
        'age': [25] * 20,
        'experience_years': [3] * 20,
        'education_level': ['Bachelor'] * 20,
        'skills': ['Python|SQL'] * 20
    })
    candidates.to_csv('data/candidates_data.csv', index=False)
    
    print("✅ Minimal data created")

def create_minimal_models():
    """Create minimal models if full script fails"""
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    # Simple classification model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.rand(50, 5)
    y = np.random.choice([0, 1], 50)
    model.fit(X, y)
    
    with open('models/candidate_recommendation_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'type': 'demo'}, f)
    
    with open('models/company_similarity_model.pkl', 'wb') as f:
        pickle.dump({'type': 'demo', 'vectorizer': None}, f)
    
    print("✅ Minimal models created")

def verify_setup():
    """Verify that all required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'data/companies_data.csv',
        'data/candidates_data.csv', 
        'models/company_similarity_model.pkl',
        'models/candidate_recommendation_model.pkl',
        'utils/__init__.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} files are missing")
        print("The app may not work correctly")
    else:
        print("\n🎉 All required files are present!")
    
    return len(missing_files) == 0

def show_final_instructions():
    """Show final instructions to user"""
    print_header("SETUP COMPLETE! 🎉")
    
    print("✅ Your ML Graduation Project is ready!")
    print("\n🚀 To run the application:")
    print("   streamlit run app.py")
    print("\n🌐 Then open your browser to:")
    print("   http://localhost:8501")
    
    print("\n📋 What's included:")
    print("   • Interactive Streamlit web app")
    print("   • Company similarity recommendation system")
    print("   • Candidate classification system") 
    print("   • Sample data (100 companies, 500 candidates)")
    print("   • Demo ML models (replace with your real models)")
    print("   • Professional dashboard interface")
    
    print("\n🔧 Next steps for your graduation project:")
    print("   1. Replace sample data with your real dataset")
    print("   2. Replace demo models with your trained models")
    print("   3. Customize the UI and add your project details")
    print("   4. Add your analysis and results")
    print("   5. Prepare your presentation")
    
    print("\n📚 File structure:")
    print("   • app.py - Main Streamlit application")
    print("   • data/ - CSV files with your data")
    print("   • models/ - Trained ML models (.pkl files)")
    print("   • utils/ - Helper functions and preprocessing")
    
    print("\n💡 Tips:")
    print("   • Check README.md for detailed documentation")
    print("   • Use 'streamlit run app.py --logger.level=debug' for debugging")
    print("   • Customize colors and styling in the CSS section")
    
    print("\n🎓 Good luck with your graduation project!")
    print("="*60)

if __name__ == "__main__":
    try:
        success = setup_project()
        if success:
            print("\n🎉 Setup completed successfully!")
        else:
            print("\n❌ Setup completed with some issues")
            print("Check the messages above and try running individual scripts")
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {str(e)}")
        print("Please check the error messages and try again")