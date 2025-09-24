#!/usr/bin/env python3
"""
Install script for Enhanced Mental Healthcare Chatbot

This script installs the required dependencies for the chatbot application.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main installation function"""
    print("🚀 Installing Enhanced Mental Healthcare Chatbot Dependencies")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Update pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Updating pip")
    
    # Install core packages
    core_packages = [
        "flask==2.3.3",
        "flask-cors==4.0.0",
        "flask-limiter==3.5.0",
        "werkzeug==2.3.7",
        "pyjwt==2.8.0",
        "cryptography==41.0.4",
        "redis==5.0.1",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.0"
    ]
    
    print("\n📦 Installing core packages...")
    for package in core_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", 
                          f"Installing {package.split('==')[0]}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Install ML/NLP packages
    ml_packages = [
        "transformers==4.35.0",
        "torch>=1.13.0",
        "sentence-transformers==2.2.2",
        "huggingface-hub==0.16.4",
        "nltk==3.8.1",
        "textblob==0.17.1",
        "spacy==3.6.1",
        "vaderSentiment==3.3.2"
    ]
    
    print("\n🧠 Installing ML/NLP packages...")
    for package in ml_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", 
                          f"Installing {package.split('>=')[0].split('==')[0]}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data...")
    run_command(f"{sys.executable} -c \"import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('stopwords')\"",
                "Downloading NLTK data")
    
    # Download spaCy model
    print("\n🌍 Installing spaCy English model...")
    run_command(f"{sys.executable} -m spacy download en_core_web_sm", 
                "Installing spaCy English model")
    
    print("\n" + "=" * 60)
    print("🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Run: python web_app.py")
    print("2. Open your browser to: http://localhost:5000")
    print("3. Start chatting with the mental health assistant!")

if __name__ == "__main__":
    main()
