#!/usr/bin/env python3
"""
Setup script for Enhanced Mental Healthcare Chatbot

This script helps set up the environment and install required packages
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package_name):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        logger.error("Python 3.7 or higher is required")
        return False
    logger.info(f"Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def install_requirements():
    """Install required packages"""
    # Essential packages (always install)
    essential_packages = [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0"
    ]
    
    # Advanced packages (install with fallback)
    advanced_packages = [
        "transformers>=4.20.0",
        "torch>=1.10.0",
        "sentence-transformers>=2.2.0",
        "nltk>=3.7",
        "textblob>=0.17.0"
    ]
    
    # Security packages
    security_packages = [
        "cryptography>=3.4.0"
    ]
    
    # Install essential packages
    logger.info("Installing essential packages...")
    for package in essential_packages:
        if not install_package(package):
            logger.error(f"Failed to install essential package: {package}")
            return False
    
    # Install advanced packages (with fallback)
    logger.info("Installing advanced packages (these may take a while)...")
    advanced_success = 0
    for package in advanced_packages:
        if install_package(package):
            advanced_success += 1
    
    if advanced_success < len(advanced_packages):
        logger.warning(f"Only {advanced_success}/{len(advanced_packages)} advanced packages installed")
        logger.warning("The chatbot will run with reduced functionality")
    
    # Install security packages
    logger.info("Installing security packages...")
    for package in security_packages:
        install_package(package)  # Non-critical, continue if failed
    
    return True

def setup_data_directory():
    """Create necessary directories"""
    directories = ["data", "logs", "models"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def download_sample_data():
    """Download or create sample data if needed"""
    if not os.path.exists("emotion.csv"):
        logger.info("Creating sample emotion data...")
        
        # Create a simple sample dataset
        sample_data = {
            'text': [
                "I feel very happy today",
                "I am sad and depressed",
                "I feel anxious about tomorrow",
                "I am excited about the future",
                "I feel hopeless",
                "I am grateful for everything",
                "I feel overwhelmed",
                "I am content with my life"
            ],
            'emotions': [
                'joy',
                'sadness',
                'fear',
                'joy',
                'sadness',
                'joy',
                'fear',
                'joy'
            ]
        }
        
        try:
            import pandas as pd
            df = pd.DataFrame(sample_data)
            df.to_csv("emotion.csv", index=False)
            logger.info("Sample emotion data created")
        except ImportError:
            logger.warning("Could not create sample data - pandas not available")

def setup_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except ImportError:
        logger.info("NLTK not available - skipping NLTK data download")

def run_system_check():
    """Run system compatibility check"""
    logger.info("Running system compatibility check...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.total < 2 * 1024 * 1024 * 1024:  # 2GB
            logger.warning("System has less than 2GB RAM - advanced features may be limited")
    except ImportError:
        logger.info("Could not check system memory")
    
    # Check internet connection for package downloads
    try:
        import urllib.request
        urllib.request.urlopen('https://pypi.org', timeout=5)
        logger.info("Internet connection available for package downloads")
    except:
        logger.warning("Limited internet connectivity - some packages may not install")
    
    return True

def main():
    """Main setup function"""
    logger.info("Starting Enhanced Mental Healthcare Chatbot Setup...")
    
    # Run system check
    if not run_system_check():
        logger.error("System compatibility check failed")
        return False
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install required packages")
        return False
    
    # Setup directories
    setup_data_directory()
    
    # Download sample data
    download_sample_data()
    
    # Setup NLTK data
    setup_nltk_data()
    
    logger.info("Setup completed successfully!")
    logger.info("\nTo run the enhanced chatbot:")
    logger.info("python chatbot.py")
    
    # Test basic imports
    logger.info("\nTesting basic imports...")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        logger.info("✓ Core packages imported successfully")
    except ImportError as e:
        logger.error(f"✗ Core package import failed: {e}")
        return False
    
    # Test advanced imports
    try:
        import transformers
        import torch
        logger.info("✓ Advanced NLP packages available")
    except ImportError:
        logger.warning("⚠ Advanced NLP packages not available - using fallback mode")
    
    logger.info("\nSetup verification completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
