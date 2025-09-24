#!/usr/bin/env python3
"""
Simplified Model Evaluation Script for Mental Healthcare Chatbot

This script provides essential evaluation metrics including:
- ROC Curves 
- Confusion Matrix
- Classification Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ChatbotModelEvaluator:
    """Simplified model evaluation for chatbot classification"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.vectorizer = None
        self.encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train_encoded = None
        self.y_test_encoded = None
        self.y_pred = None
        self.y_pred_proba = None
        self.classes = None
        
    def create_sample_data(self):
        """Create sample mental health text data for demonstration"""
        # Expanded dataset with more distinct patterns for better classification
        sample_data = {
            'text': [
                # Depression samples - clear depressive language
                "I feel so sad and hopeless today, everything is meaningless",
                "Feeling very depressed and worthless, can't stop crying",
                "I feel empty and alone, depression is consuming me",
                "Everything feels meaningless, nothing matters anymore",
                "Life has no meaning, I'm struggling with dark thoughts",
                "Can't stop crying, feel helpless and lost",
                "Depressed and worthless, everything is falling apart",
                "Feeling hollow inside, sadness overwhelming me",
                "No hope left, darkness everywhere",
                "Deeply sad, can't find joy in anything",
                "Worthless feelings consuming my thoughts",
                "Heavy heart, can't shake this depression",
                
                # Anxiety samples - clear anxiety language
                "I'm really anxious about my future, worried constantly",
                "I can't sleep, having nightmares, anxiety attacks",
                "I'm so worried about everything, stress overwhelming",
                "Anxiety is taking over my life, panic attacks frequent",
                "Nervous about meeting people, social anxiety high",
                "Feeling tense and on edge, worried sick",
                "Feeling anxious about work, stress levels through roof",
                "Worried about my health, anxiety consuming thoughts",
                "Panic attacks frequent, nervous breakdown coming",
                "Stressed and anxious, can't calm down",
                "Worried constantly, anxiety ruling my life",
                "Nervous energy, can't stop worrying",
                
                # Positive samples - clear positive language  
                "Everything seems great, I'm happy and content",
                "Life is beautiful, feeling positive and grateful",
                "Today was a wonderful day, excited about future",
                "Feeling grateful and content, life is good",
                "I'm doing really well lately, happy and optimistic",
                "Feeling optimistic about future, great things coming",
                "Happy to be alive today, blessed and grateful",
                "Excited about the future, wonderful opportunities ahead",
                "Feeling blessed and joyful, life is amazing",
                "Great mood today, everything is wonderful",
                "Happy and content, grateful for everything",
                "Optimistic and positive, life is beautiful",
                
                # Crisis samples - clear crisis language
                "I want to end my life, can't take it anymore",
                "Having suicidal thoughts, want everything to stop",
                "I feel trapped and hopeless, considering ending it",
                "Suicidal ideation strong, planning to harm myself",
                "Want to die, life not worth living anymore",
                "Contemplating suicide, no other way out",
                "Planning to end my suffering, suicide only option",
                "Hopeless and trapped, death seems like relief"
            ],
            'emotions': [
                # Depression labels
                'depression', 'depression', 'depression', 'depression',
                'depression', 'depression', 'depression', 'depression',
                'depression', 'depression', 'depression', 'depression',
                
                # Anxiety labels  
                'anxiety', 'anxiety', 'anxiety', 'anxiety',
                'anxiety', 'anxiety', 'anxiety', 'anxiety',
                'anxiety', 'anxiety', 'anxiety', 'anxiety',
                
                # Positive labels
                'positive', 'positive', 'positive', 'positive',
                'positive', 'positive', 'positive', 'positive',
                'positive', 'positive', 'positive', 'positive',
                
                # Crisis labels
                'crisis', 'crisis', 'crisis', 'crisis',
                'crisis', 'crisis', 'crisis', 'crisis'
            ]
        }
        return pd.DataFrame(sample_data)
    
    def load_data(self, data_path=None):
        """Load data from file or create sample data"""
        if data_path:
            try:
                self.data = pd.read_csv(data_path)
                print(f"✅ Data loaded successfully: {self.data.shape}")
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                print("Using sample data instead...")
                self.data = self.create_sample_data()
        else:
            self.data = self.create_sample_data()
            print("📊 Using sample data for demonstration")
        
        print(f"Target distribution:\n{self.data['emotions'].value_counts()}")
        return True
    
    def prepare_data(self, test_size=0.25, random_state=123):
        """Prepare data for training and testing"""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return False
        
        # Split the data
        train_data, test_data = train_test_split(
            self.data, test_size=test_size, random_state=random_state, 
            stratify=self.data['emotions']
        )
        
        # Prepare features and targets
        X_train_text = train_data['text']
        X_test_text = test_data['text']
        y_train = train_data['emotions']
        y_test = test_data['emotions']
        
        # Create and fit vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            max_df=0.85, 
            min_df=1,
            stop_words='english', 
            max_features=2000,
            ngram_range=(1, 2),
            analyzer='word'
        )
        self.X_train = self.vectorizer.fit_transform(X_train_text)
        self.X_test = self.vectorizer.transform(X_test_text)
        
        # Encode labels
        self.encoder = LabelEncoder()
        self.y_train_encoded = self.encoder.fit_transform(y_train)
        self.y_test_encoded = self.encoder.transform(y_test)
        self.classes = self.encoder.classes_
        
        print(f"✅ Data prepared successfully")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Classes: {list(self.classes)}")
        
        return True
    
    def train_model(self):
        """Train the logistic regression model"""
        if self.X_train is None:
            print("❌ Data not prepared. Please prepare data first.")
            return False
        
        # Train model with optimized hyperparameters
        self.model = LogisticRegression(
            C=10.0,  # Higher regularization for better performance
            class_weight='balanced',
            random_state=123,
            max_iter=2000,
            solver='lbfgs',
            multi_class='ovr'
        )
        self.model.fit(self.X_train, self.y_train_encoded)
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Calculate basic metrics
        accuracy = accuracy_score(self.y_test_encoded, self.y_pred)
        print(f"✅ Model trained successfully")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return True
    
    def plot_confusion_matrix(self, figsize=(10, 8)):
        """Plot confusion matrix heatmap"""
        if self.y_pred is None:
            print("❌ Model not trained. Please train model first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test_encoded, self.y_pred)
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix - Mental Health Classification', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Print detailed metrics per class
        print("\n📊 Confusion Matrix Analysis:")
        print("="*60)
        for i, class_name in enumerate(self.classes):
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
    
    def plot_roc_curves_multiclass(self, figsize=(12, 8)):
        """Plot ROC curves for multiclass classification using One-vs-Rest"""
        if self.y_pred_proba is None:
            print("❌ Model not trained. Please train model first.")
            return
        
        plt.figure(figsize=figsize)
        
        n_classes = len(self.classes)
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        # Calculate ROC for each class (One-vs-Rest)
        for i, (class_name, color) in enumerate(zip(self.classes, colors)):
            # Create binary labels: current class vs all others
            y_binary = (self.y_test_encoded == i).astype(int)
            y_scores = self.y_pred_proba[:, i]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_binary, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line (random classifier)
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
        
        print("\n📈 ROC-AUC Scores (One-vs-Rest):")
        print("="*50)
        for i, class_name in enumerate(self.classes):
            y_binary = (self.y_test_encoded == i).astype(int)
            y_scores = self.y_pred_proba[:, i]
            fpr, tpr, _ = roc_curve(y_binary, y_scores)
            roc_auc = auc(fpr, tpr)
            print(f"{class_name:12s}: AUC = {roc_auc:.4f}")
    
    def plot_binary_roc_crisis_detection(self, figsize=(8, 6)):
        """Plot ROC curve for crisis vs non-crisis detection"""
        if self.y_pred_proba is None:
            print("❌ Model not trained. Please train model first.")
            return
        
        # Create binary classification: crisis vs non-crisis
        crisis_idx = np.where(self.classes == 'crisis')[0]
        if len(crisis_idx) == 0:
            print("❌ No 'crisis' class found in the data.")
            return
        
        crisis_idx = crisis_idx[0]
        
        # Binary labels: 1 for crisis, 0 for non-crisis
        y_binary = (self.y_test_encoded == crisis_idx).astype(int)
        y_scores = self.y_pred_proba[:, crisis_idx]  # Probability of crisis class
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_binary, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=3,
                label=f'Crisis Detection (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8,
                label='Random Classifier')
        
        # Find optimal threshold (closest to top-left corner)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                label=f'Optimal Threshold ({optimal_threshold:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
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
        print(f"  Sensitivity (True Positive Rate): {tpr[optimal_idx]:.3f}")
        print(f"  Specificity (1 - False Positive Rate): {1-fpr[optimal_idx]:.3f}")
        
        # Performance at different thresholds
        print(f"\nPerformance at different thresholds:")
        for thresh in [0.3, 0.5, 0.7]:
            idx = np.argmin(np.abs(thresholds - thresh))
            print(f"  Threshold {thresh}: Sensitivity={tpr[idx]:.3f}, Specificity={1-fpr[idx]:.3f}")
    
    def print_classification_report(self):
        """Print detailed classification report"""
        if self.y_pred is None:
            print("❌ Model not trained. Please train model first.")
            return
        
        print("\n📋 Classification Report:")
        print("="*70)
        report = classification_report(
            self.y_test_encoded, self.y_pred, 
            target_names=self.classes,
            digits=4
        )
        print(report)
    
    def analyze_feature_importance(self, top_n=10):
        """Analyze most important features for each class"""
        if self.model is None or self.vectorizer is None:
            print("❌ Model not trained. Please train model first.")
            return
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"\n🔍 Top {top_n} Most Important Features by Class:")
        print("="*70)
        
        for i, class_name in enumerate(self.classes):
            coef = self.model.coef_[i]
            top_positive_indices = np.argsort(coef)[-top_n:][::-1]
            top_negative_indices = np.argsort(coef)[:top_n]
            
            print(f"\n{class_name.upper()} CLASS:")
            print(f"  Most Predictive Features (Positive):")
            for j, idx in enumerate(top_positive_indices):
                feature = feature_names[idx]
                weight = coef[idx]
                print(f"    {j+1:2d}. {feature:20s} ({weight:+.4f})")
            
            print(f"  Least Predictive Features (Negative):")
            for j, idx in enumerate(top_negative_indices):
                feature = feature_names[idx]
                weight = coef[idx]
                print(f"    {j+1:2d}. {feature:20s} ({weight:+.4f})")
    
    def run_complete_evaluation(self, data_path=None):
        """Run complete model evaluation pipeline"""
        print("🧠 Mental Healthcare Chatbot - Model Evaluation")
        print("="*70)
        
        # Load data
        self.load_data(data_path)
        
        # Prepare data
        if not self.prepare_data():
            return False
        
        # Train model
        if not self.train_model():
            return False
        
        # Generate all evaluations
        print("\n📊 Generating Evaluation Reports and Plots...")
        print("="*70)
        
        self.print_classification_report()
        self.plot_confusion_matrix()
        self.plot_roc_curves_multiclass()
        self.plot_binary_roc_crisis_detection()
        self.analyze_feature_importance()
        
        print("\n✅ Complete evaluation finished!")
        print("="*70)
        return True

def main():
    """Main function to run model evaluation"""
    evaluator = ChatbotModelEvaluator()
    
    # You can specify your data path here
    # data_path = "path/to/your/mental_health_data.csv"
    data_path = None  # Will use sample data
    
    # Run complete evaluation
    evaluator.run_complete_evaluation(data_path)

if __name__ == "__main__":
    main()
