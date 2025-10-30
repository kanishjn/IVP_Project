#!/usr/bin/env python3
"""
Main SVM Training Script - Astronomical Object Classification
Trains SVM on balanced dataset with train/test split
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

class AstronomicalSVMTrainer:
    """Main SVM trainer with balanced dataset and proper train/test split"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.pipeline = None
        self.le = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self):
        """Load data and create balanced dataset"""
        print("="*80)
        print("LOADING DATASET")
        print("="*80)
        
        X = np.load('features.npy')
        y_str = np.load('labels.npy')
        
        print(f"\nOriginal Dataset Statistics:")
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"\nOriginal Class Distribution:")
        for class_name, count in Counter(y_str).items():
            percentage = (count / len(y_str)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Encode labels
        self.le = LabelEncoder()
        y = self.le.fit_transform(y_str)
        
        # Balance classes by undersampling to minority class
        min_samples = min(Counter(y).values())
        print(f"\n🔄 Balancing dataset to {min_samples} samples per class...")
        
        balanced_indices = []
        for class_idx in range(len(self.le.classes_)):
            class_indices = np.where(y == class_idx)[0]
            selected_indices = np.random.choice(class_indices, min_samples, replace=False)
            balanced_indices.extend(selected_indices)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        print(f"\nBalanced Dataset Statistics:")
        print(f"  Total samples: {X_balanced.shape[0]}")
        print(f"\nBalanced Class Distribution:")
        for i, class_name in enumerate(self.le.classes_):
            count = np.sum(y_balanced == i)
            percentage = (count / len(y_balanced)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return X_balanced, y_balanced
    
    def engineer_features(self, X, verbose=True):
        """Apply advanced feature engineering"""
        if verbose:
            print("\n" + "="*80)
            print("FEATURE ENGINEERING")
            print("="*80)
        
        new_features = []
        
        # 1. Intensity features
        intensity_range = X[:, 8:9] - X[:, 9:10]
        intensity_contrast = (X[:, 8:9] - X[:, 7:8]) / (X[:, 7:8] + 1e-6)
        intensity_normalized = X[:, 7:8] / (X[:, 8:9] + 1e-6)
        new_features.extend([intensity_range, intensity_contrast, intensity_normalized])
        
        # 2. Morphological features
        compactness = X[:, 0:1] / ((X[:, 1:2] + X[:, 2:3]) + 1e-6)
        circularity = 4 * np.pi * X[:, 0:1] / ((X[:, 1:2] * X[:, 2:3]) + 1e-6)
        shape_complexity = X[:, 4:5] * X[:, 3:4]
        elongation = np.maximum(X[:, 1:2], X[:, 2:3]) / (np.minimum(X[:, 1:2], X[:, 2:3]) + 1e-6)
        new_features.extend([compactness, circularity, shape_complexity, elongation])
        
        # 3. Combined features
        area_intensity = X[:, 0:1] * X[:, 7:8]
        perimeter_approx = 2 * (X[:, 1:2] + X[:, 2:3])
        area_perimeter_ratio = X[:, 0:1] / (perimeter_approx + 1e-6)
        new_features.extend([area_intensity, area_perimeter_ratio])
        
        # 4. Log-transformed features
        log_area = np.log10(X[:, 0:1] + 1)
        log_intensity = np.log10(X[:, 7:8] + 1)
        log_max_intensity = np.log10(X[:, 8:9] + 1)
        new_features.extend([log_area, log_intensity, log_max_intensity])
        
        # 5. Radial profile analysis
        radial = X[:, 27:35]
        radial_gradients = np.diff(radial, axis=1)
        radial_mean = np.mean(radial, axis=1, keepdims=True)
        radial_std = np.std(radial, axis=1, keepdims=True)
        radial_max_min_ratio = (np.max(radial, axis=1, keepdims=True) / 
                                (np.min(radial, axis=1, keepdims=True) + 1e-6))
        new_features.extend([radial_gradients, radial_mean, radial_std, radial_max_min_ratio])
        
        # 6. Hu moments statistics
        hu_moments = X[:, 10:17]
        hu_mean = np.mean(hu_moments, axis=1, keepdims=True)
        hu_std = np.std(hu_moments, axis=1, keepdims=True)
        new_features.extend([hu_mean, hu_std])
        
        # 7. LBP texture statistics
        lbp = X[:, 17:27]
        lbp_entropy = -np.sum(lbp * np.log(lbp + 1e-10), axis=1, keepdims=True)
        lbp_uniformity = np.sum(lbp ** 2, axis=1, keepdims=True)
        new_features.extend([lbp_entropy, lbp_uniformity])
        
        X_engineered = np.hstack([X] + new_features)
        
        if verbose:
            print(f"  Original features: {X.shape[1]}")
            print(f"  Total features after engineering: {X_engineered.shape[1]}")
            print(f"  New features added: {X_engineered.shape[1] - X.shape[1]}")
        
        return X_engineered
    
    def split_data(self, X, y):
        """Split data into train and test sets"""
        print("\n" + "="*80)
        print("TRAIN/TEST SPLIT")
        print("="*80)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"\nSplit Information:")
        print(f"  Training samples: {self.X_train.shape[0]} ({self.X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test samples: {self.X_test.shape[0]} ({self.X_test.shape[0]/len(X)*100:.1f}%)")
        
        print(f"\nTraining Set Class Distribution:")
        for i, class_name in enumerate(self.le.classes_):
            count = np.sum(self.y_train == i)
            percentage = (count / len(self.y_train)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\nTest Set Class Distribution:")
        for i, class_name in enumerate(self.le.classes_):
            count = np.sum(self.y_test == i)
            percentage = (count / len(self.y_test)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def train_model(self, C=10, gamma=0.05, kernel='rbf', n_features=50):
        """Train SVM model with specified parameters"""
        print("\n" + "="*80)
        print("TRAINING SVM MODEL")
        print("="*80)
        
        print(f"\nModel Parameters:")
        print(f"  Kernel: {kernel}")
        print(f"  C: {C}")
        print(f"  Gamma: {gamma}")
        print(f"  Features selected: {n_features}")
        print(f"  Class weight: balanced")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(mutual_info_classif, k=n_features)),
            ('svm', SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced', 
                       random_state=self.random_state))
        ])
        
        # Train
        print("\n⏳ Training model...")
        start_time = time.time()
        self.pipeline.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        # Cross-validation on training set
        print("\n⏳ Performing 5-fold cross-validation on training set...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.pipeline, self.X_train, self.y_train, 
                                    cv=cv, scoring='f1_weighted', n_jobs=-1)
        
        print(f"\nCross-Validation Results:")
        print(f"  F1 Scores: {cv_scores}")
        print(f"  Mean F1: {cv_scores.mean():.4f}")
        print(f"  Std Dev: {cv_scores.std():.4f}")
        
        return cv_scores.mean(), cv_scores.std()
    
    def evaluate_test_set(self):
        """Evaluate model on test set"""
        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)
        
        # Predictions
        y_pred = self.pipeline.predict(self.X_test)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        
        print(f"\nOverall Test Performance:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1 Score (weighted): {f1_weighted:.4f}")
        print(f"  F1 Score (macro): {f1_macro:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.le.classes_, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Per-class analysis
        print("\nPer-Class Analysis:")
        for i, class_name in enumerate(self.le.classes_):
            class_mask = (self.y_test == i)
            class_total = np.sum(class_mask)
            class_correct = np.sum((self.y_test[class_mask] == y_pred[class_mask]))
            class_accuracy = class_correct / class_total if class_total > 0 else 0
            print(f"  {class_name}: {class_correct}/{class_total} ({class_accuracy:.1%})")
        
        return accuracy, f1_weighted, cm, y_pred
    
    def save_model(self, filename='balanced_svm_model.joblib'):
        """Save trained model and metadata"""
        print(f"\n💾 Saving model to '{filename}'...")
        
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.le,
            'train_size': self.X_train.shape[0],
            'test_size': self.X_test.shape[0],
            'feature_names': self.get_feature_names(),
            'class_names': list(self.le.classes_),
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filename)
        print("✓ Model saved successfully!")
    
    def get_feature_names(self):
        """Generate feature names"""
        base_features = [
            'area', 'major_axis', 'minor_axis', 'eccentricity', 'solidity',
            'extent', 'orientation', 'mean_intensity', 'max_intensity', 'min_intensity'
        ]
        base_features.extend([f'hu_{i+1}' for i in range(7)])
        base_features.extend([f'lbp_{i+1}' for i in range(10)])
        base_features.extend([f'radial_{i+1}' for i in range(8)])
        
        engineered_features = [
            'intensity_range', 'intensity_contrast', 'intensity_normalized',
            'compactness', 'circularity', 'shape_complexity', 'elongation',
            'area_intensity', 'area_perimeter_ratio',
            'log_area', 'log_intensity', 'log_max_intensity'
        ]
        engineered_features.extend([f'radial_grad_{i+1}' for i in range(7)])
        engineered_features.extend(['radial_mean', 'radial_std', 'radial_max_min_ratio'])
        engineered_features.extend(['hu_mean', 'hu_std', 'lbp_entropy', 'lbp_uniformity'])
        
        return base_features + engineered_features
    
    def plot_confusion_matrix(self, cm, filename='confusion_matrix.png'):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.le.classes_,
                    yticklabels=self.le.classes_,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Balanced SVM', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to '{filename}'")
        plt.close()

def main():
    print("="*80)
    print("ASTRONOMICAL OBJECT CLASSIFICATION - SVM TRAINING")
    print("="*80)
    print("\nObjective: Train SVM to classify GALAXY, QSO, and STAR objects")
    print("Approach: Balanced dataset with train/test split")
    print()
    
    # Initialize trainer
    trainer = AstronomicalSVMTrainer(random_state=42)
    
    # Load and balance data
    X, y = trainer.load_and_prepare_data()
    
    # Engineer features
    X_eng = trainer.engineer_features(X)
    
    # Split data
    trainer.split_data(X_eng, y)
    
    # Train model (using best known parameters)
    cv_mean, cv_std = trainer.train_model(C=10, gamma=0.05, kernel='rbf', n_features=50)
    
    # Evaluate on test set
    test_acc, test_f1, cm, y_pred = trainer.evaluate_test_set()
    
    # Save model
    trainer.save_model('balanced_svm_model.joblib')
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(cm, 'balanced_svm_confusion_matrix.png')
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"\n📊 Cross-Validation (Training Set):")
    print(f"   Mean F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"\n📊 Test Set Performance:")
    print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   F1 Score: {test_f1:.4f}")
    print(f"\n✅ Files Created:")
    print(f"   - balanced_svm_model.joblib")
    print(f"   - balanced_svm_confusion_matrix.png")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
