#!/usr/bin/env python3
"""
ROC Curve and Confusion Matrix Demo for Mental Healthcare Chatbot

This script demonstrates how to generate ROC curves and confusion matrix 
for your chatbot's classification model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample mental health text data"""
    data = {
        'text': [
            "I feel so sad and hopeless today", "I'm really anxious about my future",
            "Everything seems great, I'm happy", "I can't sleep, having nightmares",
            "Feeling very depressed and worthless", "I'm so worried about everything",
            "Life is beautiful, feeling positive", "Having panic attacks frequently",
            "I feel empty and alone", "Stress is overwhelming me",
            "Today was a wonderful day", "Can't stop crying, feel helpless",
            "Anxiety is taking over my life", "Feeling grateful and content",
            "Everything feels meaningless", "Nervous about meeting people",
            "I'm doing really well lately", "Feeling tense and on edge",
            "Depression is consuming me", "Feeling optimistic about future",
            "I want to end my life", "Having suicidal thoughts",
            "Nothing matters anymore", "I feel trapped and hopeless",
            "Life has no meaning", "I'm struggling with dark thoughts",
            "Feeling anxious about work", "Happy to be alive today",
            "Worried about my health", "Excited about the future"
        ],
        'emotions': [
            'depression', 'anxiety', 'positive', 'anxiety',
            'depression', 'anxiety', 'positive', 'anxiety',
            'depression', 'anxiety', 'positive', 'depression',
            'anxiety', 'positive', 'depression', 'anxiety',
            'positive', 'anxiety', 'depression', 'positive',
            'crisis', 'crisis', 'depression', 'crisis',
            'crisis', 'depression', 'anxiety', 'positive',
            'anxiety', 'positive'
        ]
    }
    return pd.DataFrame(data)

def train_model():
    """Train a simple classification model"""
    # Load sample data
    data = create_sample_data()
    print("🧠 Mental Healthcare Chatbot - Model Evaluation")
    print("="*60)
    print(f"Data loaded: {data.shape[0]} samples")
    print(f"Class distribution:\n{data['emotions'].value_counts()}")
    
    # Split data
    train_data, test_data = train_test_split(
        data, test_size=0.3, random_state=42, stratify=data['emotions']
    )
    
    # Prepare features
    vectorizer = TfidfVectorizer(max_df=0.9, stop_words='english', max_features=1000)
    X_train = vectorizer.fit_transform(train_data['text'])
    X_test = vectorizer.transform(test_data['text'])
    
    # Prepare labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_data['emotions'])
    y_test = encoder.transform(test_data['emotions'])
    classes = encoder.classes_
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {list(classes)}")
    
    # Train model
    model = LogisticRegression(C=0.1, class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model trained successfully!")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return model, vectorizer, encoder, X_test, y_test, y_pred, y_pred_proba, classes

def plot_confusion_matrix(y_test, y_pred, classes):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Mental Health Classification', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Print metrics per class
    print("\n📊 Confusion Matrix Analysis:")
    print("="*60)
    for i, class_name in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{class_name.upper()}:")
        print(f"  True Positives: {tp:2d}   False Positives: {fp:2d}")
        print(f"  False Negatives: {fn:2d}   True Negatives: {tn:2d}")
        print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")

def plot_roc_curves(y_test, y_pred_proba, classes):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot ROC curve for each class (One-vs-Rest)
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        # Create binary labels: current class vs all others
        y_binary = (y_test == i).astype(int)
        y_scores = y_pred_proba[:, i]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_binary, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Mental Health Classification (One-vs-Rest)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print AUC scores
    print("\n📈 ROC-AUC Scores (One-vs-Rest):")
    print("="*50)
    for i, class_name in enumerate(classes):
        y_binary = (y_test == i).astype(int)
        y_scores = y_pred_proba[:, i]
        fpr, tpr, _ = roc_curve(y_binary, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"{class_name:12s}: AUC = {roc_auc:.4f}")

def plot_crisis_roc(y_test, y_pred_proba, classes):
    """Plot ROC curve specifically for crisis detection"""
    # Find crisis class index
    try:
        crisis_idx = list(classes).index('crisis')
    except ValueError:
        print("❌ No 'crisis' class found")
        return
    
    plt.figure(figsize=(8, 6))
    
    # Binary classification: crisis vs non-crisis
    y_binary = (y_test == crisis_idx).astype(int)
    y_scores = y_pred_proba[:, crisis_idx]
    
    # Calculate ROC
    fpr, tpr, thresholds = roc_curve(y_binary, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=3,
            label=f'Crisis Detection (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8,
            label='Random Classifier')
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
            label=f'Optimal Threshold ({optimal_threshold:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Crisis Detection', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n🚨 Crisis Detection Analysis:")
    print("="*50)
    print(f"AUC Score: {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"At optimal threshold:")
    print(f"  Sensitivity: {tpr[optimal_idx]:.3f}")
    print(f"  Specificity: {1-fpr[optimal_idx]:.3f}")

def print_classification_report(y_test, y_pred, classes):
    """Print classification report"""
    print("\n📋 Classification Report:")
    print("="*70)
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)
    print(report)

def analyze_feature_importance(model, vectorizer, classes, top_n=10):
    """Analyze most important features"""
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\n🔍 Top {top_n} Most Important Features by Class:")
    print("="*70)
    
    for i, class_name in enumerate(classes):
        coef = model.coef_[i]
        top_indices = np.argsort(np.abs(coef))[-top_n:][::-1]
        
        print(f"\n{class_name.upper()}:")
        for j, idx in enumerate(top_indices):
            feature = feature_names[idx]
            weight = coef[idx]
            print(f"  {j+1:2d}. {feature:15s} ({weight:+.4f})")

def main():
    """Main evaluation function"""
    # Train model
    model, vectorizer, encoder, X_test, y_test, y_pred, y_pred_proba, classes = train_model()
    
    # Generate evaluations
    print("\n📊 Generating Evaluation Reports and Plots...")
    print("="*60)
    
    # Classification report
    print_classification_report(y_test, y_pred, classes)
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes)
    
    # ROC curves
    plot_roc_curves(y_test, y_pred_proba, classes)
    
    # Crisis-specific ROC
    plot_crisis_roc(y_test, y_pred_proba, classes)
    
    # Feature importance
    analyze_feature_importance(model, vectorizer, classes)
    
    print("\n✅ Model evaluation completed!")
    print("="*60)
    print("🎯 Key Insights:")
    print("   • Confusion Matrix shows prediction accuracy per class")
    print("   • ROC Curves show discrimination ability (AUC > 0.5 is better than random)")
    print("   • Crisis Detection ROC is crucial for safety - high sensitivity preferred")
    print("   • Feature importance reveals which words drive classifications")
    print("\n💡 For your actual chatbot:")
    print("   • Use your real training data instead of sample data")
    print("   • Tune the crisis detection threshold based on safety requirements")
    print("   • Monitor these metrics regularly as you collect more data")

if __name__ == "__main__":
    main()
