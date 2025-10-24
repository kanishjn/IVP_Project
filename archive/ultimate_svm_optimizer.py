#!/usr/bin/env python3
"""
Ultimate SVM Optimizer - Comprehensive Testing & Optimization
This script performs extensive testing, hyperparameter analysis, and outputs the best model
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import (train_test_split, GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score, cross_validate)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            f1_score, precision_score, recall_score, make_scorer)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE
from scipy.stats import randint, uniform
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

class UltimateSVMOptimizer:
    """Ultimate SVM optimizer with comprehensive testing and analysis"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.best_model = None
        self.best_score = 0
        self.all_results = []
        self.test_X = None
        self.test_y = None
        self.best_selector = None
        
    def load_and_prepare_data(self):
        """Load and prepare balanced dataset"""
        print("="*80)
        print("LOADING AND PREPARING DATA")
        print("="*80)
        
        X = np.load('features.npy')
        y_str = np.load('labels.npy')
        
        print(f"\nOriginal data:")
        print(f"  Shape: {X.shape}")
        print(f"  Class distribution: {Counter(y_str)}")
        
        # Balance dataset
        balanced_indices = []
        samples_per_class = 222  # Smallest class
        
        for class_name in np.unique(y_str):
            class_idx = np.where(y_str == class_name)[0]
            selected = np.random.choice(class_idx, size=samples_per_class, replace=False)
            balanced_indices.extend(selected)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_str_balanced = y_str[balanced_indices]
        
        self.le = LabelEncoder()
        y_balanced = self.le.fit_transform(y_str_balanced)
        
        print(f"\nBalanced data:")
        print(f"  Shape: {X_balanced.shape}")
        print(f"  Class distribution: {Counter(y_str_balanced)}")
        
        self.X_original = X_balanced
        self.y = y_balanced
        self.y_str = y_str_balanced
        
        return X_balanced, y_balanced
    
    def engineer_features(self, X, verbose=True):
        """Advanced feature engineering"""
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
            print(f"  Engineered features: {X_engineered.shape[1]}")
            print(f"  New features added: {X_engineered.shape[1] - X.shape[1]}")
        
        return X_engineered
    
    def test_current_model(self):
        """Test the current balanced_optimized model"""
        print("\n" + "="*80)
        print("TESTING CURRENT MODEL")
        print("="*80)
        
        try:
            model_data = joblib.load('ultimate_svm_model.joblib')
            pipeline = model_data['pipeline']
            
            # Prepare test data
            X_eng = self.engineer_features(self.X_original, verbose=False)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
            
            scoring = {
                'accuracy': 'accuracy',
                'f1_macro': 'f1_macro',
                'f1_weighted': 'f1_weighted',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro'
            }
            
            cv_results = cross_validate(pipeline, X_eng, self.y, cv=cv, 
                                       scoring=scoring, n_jobs=-1)
            
            print("\nCurrent Model 10-Fold Cross-Validation:")
            for metric, scores in cv_results.items():
                if metric.startswith('test_'):
                    name = metric.replace('test_', '')
                    print(f"  {name:20s}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
            
            return np.mean(cv_results['test_f1_macro'])
            
        except FileNotFoundError:
            print("  Current model not found. Will create new optimal model.")
            return 0.0
    
    def hyperparameter_analysis(self, X, y):
        """Comprehensive hyperparameter grid search"""
        print("\n" + "="*80)
        print("HYPERPARAMETER ANALYSIS - EXTENSIVE SEARCH")
        print("="*80)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        self.test_X = X_test
        self.test_y = y_test
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Test different strategies
        strategies = [
            {
                'name': 'RBF Kernel - Extensive',
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(probability=True, random_state=self.random_state))
                ]),
                'params': {
                    'svm__C': [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
                    'svm__gamma': ['scale', 'auto', 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    'svm__kernel': ['rbf'],
                    'svm__class_weight': ['balanced', None]
                }
            },
            {
                'name': 'Polynomial Kernel',
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(probability=True, random_state=self.random_state))
                ]),
                'params': {
                    'svm__C': [0.5, 1, 5, 10, 50, 100],
                    'svm__degree': [2, 3, 4],
                    'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'svm__coef0': [0.0, 0.5, 1.0],
                    'svm__kernel': ['poly'],
                    'svm__class_weight': ['balanced']
                }
            },
            {
                'name': 'Linear Kernel',
                'pipeline': Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(kernel='linear', probability=True, random_state=self.random_state))
                ]),
                'params': {
                    'svm__C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
                    'svm__class_weight': ['balanced', None]
                }
            },
            {
                'name': 'RobustScaler + RBF',
                'pipeline': Pipeline([
                    ('scaler', RobustScaler()),
                    ('svm', SVC(probability=True, random_state=self.random_state))
                ]),
                'params': {
                    'svm__C': [1, 5, 10, 50, 100],
                    'svm__gamma': ['scale', 0.001, 0.01, 0.1],
                    'svm__kernel': ['rbf'],
                    'svm__class_weight': ['balanced']
                }
            }
        ]
        
        best_strategy = None
        best_strategy_score = 0
        
        for strategy in strategies:
            print(f"\n--- Testing {strategy['name']} ---")
            print(f"   Parameter combinations: {np.prod([len(v) for v in strategy['params'].values()])}")
            
            start_time = time.time()
            
            grid = GridSearchCV(
                strategy['pipeline'],
                strategy['params'],
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=0
            )
            
            grid.fit(X_train, y_train)
            elapsed = time.time() - start_time
            
            # Test performance
            y_pred = grid.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='macro')
            
            print(f"   Best CV F1: {grid.best_score_:.4f}")
            print(f"   Test Accuracy: {test_acc:.4f}")
            print(f"   Test F1: {test_f1:.4f}")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Best params: {grid.best_params_}")
            
            result = {
                'strategy': strategy['name'],
                'model': grid.best_estimator_,
                'cv_f1': grid.best_score_,
                'test_acc': test_acc,
                'test_f1': test_f1,
                'params': grid.best_params_,
                'time': elapsed
            }
            
            self.all_results.append(result)
            
            if test_f1 > best_strategy_score:
                best_strategy_score = test_f1
                best_strategy = result
                self.best_model = grid.best_estimator_
                self.best_score = test_f1
        
        return best_strategy
    
    def feature_selection_optimization(self, X, y):
        """Test with feature selection"""
        print("\n" + "="*80)
        print("FEATURE SELECTION OPTIMIZATION")
        print("="*80)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Test different feature counts
        feature_counts = [25, 30, 35, 40, 45, 50]
        
        for n_features in feature_counts:
            if n_features >= X.shape[1]:
                continue
                
            print(f"\n--- Testing with {n_features} features ---")
            
            # Create pipeline with feature selection included
            pipeline = Pipeline([
                ('selector', SelectKBest(score_func=mutual_info_classif, k=n_features)),
                ('scaler', StandardScaler()),
                ('svm', SVC(C=10, gamma=0.05, kernel='rbf', class_weight='balanced',
                           probability=True, random_state=self.random_state))
            ])
            
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            test_f1 = f1_score(y_test, y_pred, average='macro')
            
            print(f"   Test F1: {test_f1:.4f}")
            
            result = {
                'strategy': f'Feature Selection ({n_features} features)',
                'model': pipeline,
                'test_f1': test_f1,
                'n_features': n_features,
                'selector': pipeline.named_steps['selector']
            }
            
            self.all_results.append(result)
            
            if test_f1 > self.best_score:
                self.best_score = test_f1
                self.best_model = pipeline
                self.best_selector = pipeline.named_steps['selector']
    
    def final_evaluation(self):
        """Comprehensive evaluation of best model"""
        print("\n" + "="*80)
        print("FINAL EVALUATION OF BEST MODEL")
        print("="*80)
        
        # Re-engineer features for full dataset
        X_eng = self.engineer_features(self.X_original, verbose=False)
        
        # 10-fold cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        scoring = {
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro'
        }
        
        cv_results = cross_validate(self.best_model, X_eng, self.y, cv=cv,
                                    scoring=scoring, n_jobs=-1, return_train_score=True)
        
        print("\n10-Fold Cross-Validation Results:")
        print(f"  {'Metric':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print(f"  {'-'*65}")
        
        for metric in ['test_accuracy', 'test_f1_macro', 'test_f1_weighted', 
                      'test_precision_macro', 'test_recall_macro']:
            scores = cv_results[metric]
            name = metric.replace('test_', '')
            print(f"  {name:<25} {np.mean(scores):<10.4f} {np.std(scores):<10.4f} "
                  f"{np.min(scores):<10.4f} {np.max(scores):<10.4f}")
        
        # Test set evaluation
        if self.test_X is not None:
            print("\n\nTest Set Performance:")
            y_pred = self.best_model.predict(self.test_X)
            y_pred_str = self.le.inverse_transform(y_pred)
            y_test_str = self.le.inverse_transform(self.test_y)
            
            print(classification_report(y_test_str, y_pred_str))
            
            # Confusion matrix
            cm = confusion_matrix(y_test_str, y_pred_str, labels=self.le.classes_)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                       xticklabels=self.le.classes_, yticklabels=self.le.classes_,
                       cbar_kws={'label': 'Count'})
            plt.title(f'Ultimate SVM Model - Confusion Matrix\nTest F1: {self.best_score:.4f}',
                     fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            plt.savefig('ultimate_svm_confusion_matrix.png', dpi=150, bbox_inches='tight')
            print("\n✓ Confusion matrix saved: ultimate_svm_confusion_matrix.png")
            plt.show()
    
    def generate_comprehensive_report(self):
        """Generate detailed analysis report"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # Sort results
        sorted_results = sorted(self.all_results, key=lambda x: x.get('test_f1', 0), reverse=True)
        
        report = []
        report.append("="*80)
        report.append("ULTIMATE SVM OPTIMIZATION - COMPREHENSIVE REPORT")
        report.append("="*80)
        report.append("")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Random Seed: {self.random_state}")
        report.append(f"Dataset: {len(self.y)} balanced samples")
        report.append(f"Classes: {list(self.le.classes_)}")
        report.append("")
        
        report.append("="*80)
        report.append("ALL STRATEGIES TESTED (Ranked by Test F1)")
        report.append("="*80)
        report.append("")
        
        for i, result in enumerate(sorted_results[:10]):  # Top 10
            report.append(f"{i+1}. {result['strategy']}")
            if 'test_f1' in result:
                report.append(f"   Test F1: {result['test_f1']:.4f}")
            if 'test_acc' in result:
                report.append(f"   Test Accuracy: {result['test_acc']:.4f}")
            if 'cv_f1' in result:
                report.append(f"   CV F1: {result['cv_f1']:.4f}")
            if 'params' in result:
                report.append(f"   Best Parameters:")
                for key, val in result['params'].items():
                    report.append(f"     {key}: {val}")
            if i == 0:
                report.append("   >>> BEST MODEL <<<")
            report.append("")
        
        report.append("="*80)
        report.append("BEST MODEL DETAILS")
        report.append("="*80)
        report.append("")
        report.append(f"Strategy: {sorted_results[0]['strategy']}")
        report.append(f"Test F1 Score: {self.best_score:.4f}")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open('ultimate_svm_report.txt', 'w') as f:
            f.write(report_text)
        
        print("\n✓ Report saved: ultimate_svm_report.txt")
        print("\n" + report_text)
    
    def save_ultimate_model(self):
        """Save the ultimate optimized model"""
        print("\n" + "="*80)
        print("SAVING ULTIMATE MODEL")
        print("="*80)
        
        model_data = {
            'pipeline': self.best_model,
            'label_encoder': self.le,
            'classes': self.le.classes_,
            'test_f1_score': self.best_score,
            'random_state': self.random_state,
            'training_method': 'ultimate_optimization',
            'feature_engineering': True,
            'all_results': self.all_results
        }
        
        joblib.dump(model_data, 'ultimate_svm_model.joblib')
        print("\n✓ Model saved: ultimate_svm_model.joblib")
        print(f"✓ Test F1 Score: {self.best_score:.4f}")
    
    def run_ultimate_optimization(self):
        """Run complete optimization pipeline"""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("ULTIMATE SVM OPTIMIZER")
        print("Comprehensive Testing, Analysis, and Optimization")
        print("="*80)
        
        # 1. Load data
        X, y = self.load_and_prepare_data()
        
        # 2. Test current model
        current_score = self.test_current_model()
        
        # 3. Engineer features
        X_eng = self.engineer_features(X)
        
        # 4. Hyperparameter analysis
        best_strategy = self.hyperparameter_analysis(X_eng, y)
        
        # 5. Feature selection optimization
        self.feature_selection_optimization(X_eng, y)
        
        # 6. Final evaluation
        self.final_evaluation()
        
        # 7. Generate report
        self.generate_comprehensive_report()
        
        # 8. Save model
        self.save_ultimate_model()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE!")
        print("="*80)
        print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
        print(f"📊 Current model F1: {current_score:.4f}")
        print(f"🎯 Best model F1: {self.best_score:.4f}")
        print(f"📈 Improvement: {(self.best_score - current_score):.4f} ({(self.best_score/current_score - 1)*100:.1f}%)")
        print(f"\n💾 Files generated:")
        print(f"   ✓ ultimate_svm_model.joblib")
        print(f"   ✓ ultimate_svm_report.txt")
        print(f"   ✓ ultimate_svm_confusion_matrix.png")
        
        return self.best_model

def main():
    """Main execution"""
    optimizer = UltimateSVMOptimizer(random_state=42)
    best_model = optimizer.run_ultimate_optimization()
    return optimizer

if __name__ == "__main__":
    optimizer = main()

