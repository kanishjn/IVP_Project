#!/usr/bin/env python3
"""
SVM Hyperparameter Optimizer with Train/Test Split
Finds optimal SVM parameters using grid search on training set
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            f1_score, make_scorer)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from collections import Counter
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

class SVMOptimizer:
    """Hyperparameter optimization for SVM with proper train/test split"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.le = None
        self.best_pipeline = None
        self.grid_search = None
        
    def load_and_balance_data(self):
        """Load and balance dataset"""
        print("="*80)
        print("LOADING AND BALANCING DATA")
        print("="*80)
        
        X = np.load('features.npy')
        y_str = np.load('labels.npy')
        
        print(f"\nOriginal Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Encode labels
        self.le = LabelEncoder()
        y = self.le.fit_transform(y_str)
        
        # Balance classes
        min_samples = min(Counter(y).values())
        print(f"Balancing to {min_samples} samples per class...")
        
        balanced_indices = []
        for class_idx in range(len(self.le.classes_)):
            class_indices = np.where(y == class_idx)[0]
            selected_indices = np.random.choice(class_indices, min_samples, replace=False)
            balanced_indices.extend(selected_indices)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        print(f"✓ Balanced dataset: {X_balanced.shape[0]} samples")
        
        return X_balanced, y_balanced
    
    def engineer_features(self, X):
        """Apply feature engineering"""
        print("\nEngineering features...")
        
        new_features = []
        
        # Intensity features
        intensity_range = X[:, 8:9] - X[:, 9:10]
        intensity_contrast = (X[:, 8:9] - X[:, 7:8]) / (X[:, 7:8] + 1e-6)
        intensity_normalized = X[:, 7:8] / (X[:, 8:9] + 1e-6)
        new_features.extend([intensity_range, intensity_contrast, intensity_normalized])
        
        # Morphological features
        compactness = X[:, 0:1] / ((X[:, 1:2] + X[:, 2:3]) + 1e-6)
        circularity = 4 * np.pi * X[:, 0:1] / ((X[:, 1:2] * X[:, 2:3]) + 1e-6)
        shape_complexity = X[:, 4:5] * X[:, 3:4]
        elongation = np.maximum(X[:, 1:2], X[:, 2:3]) / (np.minimum(X[:, 1:2], X[:, 2:3]) + 1e-6)
        new_features.extend([compactness, circularity, shape_complexity, elongation])
        
        # Combined features
        area_intensity = X[:, 0:1] * X[:, 7:8]
        perimeter_approx = 2 * (X[:, 1:2] + X[:, 2:3])
        area_perimeter_ratio = X[:, 0:1] / (perimeter_approx + 1e-6)
        new_features.extend([area_intensity, area_perimeter_ratio])
        
        # Log-transformed features
        log_area = np.log10(X[:, 0:1] + 1)
        log_intensity = np.log10(X[:, 7:8] + 1)
        log_max_intensity = np.log10(X[:, 8:9] + 1)
        new_features.extend([log_area, log_intensity, log_max_intensity])
        
        # Radial profile analysis
        radial = X[:, 27:35]
        radial_gradients = np.diff(radial, axis=1)
        radial_mean = np.mean(radial, axis=1, keepdims=True)
        radial_std = np.std(radial, axis=1, keepdims=True)
        radial_max_min_ratio = (np.max(radial, axis=1, keepdims=True) / 
                                (np.min(radial, axis=1, keepdims=True) + 1e-6))
        new_features.extend([radial_gradients, radial_mean, radial_std, radial_max_min_ratio])
        
        # Hu moments statistics
        hu_moments = X[:, 10:17]
        hu_mean = np.mean(hu_moments, axis=1, keepdims=True)
        hu_std = np.std(hu_moments, axis=1, keepdims=True)
        new_features.extend([hu_mean, hu_std])
        
        # LBP texture statistics
        lbp = X[:, 17:27]
        lbp_entropy = -np.sum(lbp * np.log(lbp + 1e-10), axis=1, keepdims=True)
        lbp_uniformity = np.sum(lbp ** 2, axis=1, keepdims=True)
        new_features.extend([lbp_entropy, lbp_uniformity])
        
        X_engineered = np.hstack([X] + new_features)
        print(f"✓ Features: {X.shape[1]} → {X_engineered.shape[1]}")
        
        return X_engineered
    
    def optimize(self, X_train, y_train, quick=False):
        """Perform grid search optimization"""
        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        
        # Define parameter grid
        if quick:
            print("\n🚀 Quick optimization mode")
            param_grid = {
                'feature_selection__k': [40, 50],
                'svm__C': [1, 10],
                'svm__gamma': [0.01, 0.05],
                'svm__kernel': ['rbf']
            }
        else:
            print("\n🔬 Comprehensive optimization mode")
            param_grid = {
                'feature_selection__k': [30, 40, 50, 60],
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 0.001, 0.01, 0.05, 0.1, 1],
                'svm__kernel': ['rbf', 'poly', 'sigmoid']
            }
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(mutual_info_classif)),
            ('svm', SVC(class_weight='balanced', random_state=self.random_state))
        ])
        
        # Calculate total combinations
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)
        
        print(f"\nGrid Search Configuration:")
        print(f"  Parameter combinations: {total_combinations}")
        print(f"  Cross-validation folds: 5")
        print(f"  Total fits: {total_combinations * 5}")
        
        # Grid search
        self.grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )
        
        print(f"\n⏳ Starting grid search...")
        start_time = time.time()
        
        self.grid_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Grid search completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        # Best parameters
        print("\n" + "="*80)
        print("BEST PARAMETERS FOUND")
        print("="*80)
        
        for param, value in self.grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest CV F1 Score: {self.grid_search.best_score_:.4f}")
        
        self.best_pipeline = self.grid_search.best_estimator_
        
        return self.grid_search.best_params_, self.grid_search.best_score_
    
    def evaluate_on_test(self, X_test, y_test):
        """Evaluate best model on test set"""
        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)
        
        y_pred = self.best_pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1 Score (weighted): {f1_weighted:.4f}")
        print(f"  F1 Score (macro): {f1_macro:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.le.classes_, digits=4))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return accuracy, f1_weighted, cm
    
    def save_results(self, filename='optimized_svm_model.joblib'):
        """Save optimized model"""
        print(f"\n💾 Saving optimized model to '{filename}'...")
        
        model_data = {
            'pipeline': self.best_pipeline,
            'label_encoder': self.le,
            'best_params': self.grid_search.best_params_,
            'best_cv_score': self.grid_search.best_score_,
            'cv_results': self.grid_search.cv_results_,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filename)
        print("✓ Model saved successfully!")
    
    def display_top_configurations(self, n=10):
        """Display top N parameter configurations"""
        print(f"\n" + "="*80)
        print(f"TOP {n} CONFIGURATIONS")
        print("="*80)
        
        results = self.grid_search.cv_results_
        indices = np.argsort(results['mean_test_score'])[::-1][:n]
        
        for i, idx in enumerate(indices, 1):
            print(f"\n#{i} - F1: {results['mean_test_score'][idx]:.4f} (±{results['std_test_score'][idx]:.4f})")
            params = results['params'][idx]
            for param, value in params.items():
                print(f"    {param}: {value}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize SVM hyperparameters')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick optimization with reduced parameter grid')
    args = parser.parse_args()
    
    print("="*80)
    print("SVM HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Initialize optimizer
    optimizer = SVMOptimizer(random_state=42)
    
    # Load and balance data
    X, y = optimizer.load_and_balance_data()
    
    # Engineer features
    X_eng = optimizer.engineer_features(X)
    
    # Split data
    print("\n" + "="*80)
    print("TRAIN/TEST SPLIT")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Optimize on training set
    best_params, best_cv_score = optimizer.optimize(X_train, y_train, quick=args.quick)
    
    # Display top configurations
    optimizer.display_top_configurations(n=10)
    
    # Evaluate on test set
    test_acc, test_f1, cm = optimizer.evaluate_on_test(X_test, y_test)
    
    # Save results
    optimizer.save_results('optimized_svm_model.joblib')
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"\n✅ Best CV F1 Score: {best_cv_score:.4f}")
    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ Test F1 Score: {test_f1:.4f}")
    print(f"\n📁 Model saved to: optimized_svm_model.joblib")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
