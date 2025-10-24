#!/usr/bin/env python3
"""
ULTIMATE SVM Training Script - Maximum Performance
Combines all advanced techniques for best possible accuracy
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold, RFECV
from sklearn.decomposition import PCA
from collections import Counter
from scipy.stats import skew, kurtosis
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

class UltimateSVMTrainer:
    """Ultimate SVM trainer with maximum optimization"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.pipeline = None
        self.le = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self, use_smote=True):
        """Load data with advanced balancing"""
        print("="*80)
        print("LOADING DATASET - ULTIMATE VERSION")
        print("="*80)
        
        X = np.load('features.npy')
        y_str = np.load('labels.npy')
        
        print(f"\nOriginal Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Encode labels
        self.le = LabelEncoder()
        y = self.le.fit_transform(y_str)
        
        # Balance classes - use all STAR samples (minority class)
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
        
        print(f"\nBalanced Dataset: {X_balanced.shape[0]} samples")
        
        return X_balanced, y_balanced, use_smote
    
    def engineer_ultimate_features(self, X, verbose=True):
        """Ultimate feature engineering"""
        if verbose:
            print("\n" + "="*80)
            print("ULTIMATE FEATURE ENGINEERING")
            print("="*80)
        
        new_features = []
        
        # === INTENSITY FEATURES ===
        intensity_range = X[:, 8:9] - X[:, 9:10]
        intensity_contrast = (X[:, 8:9] - X[:, 7:8]) / (X[:, 7:8] + 1e-6)
        intensity_normalized = X[:, 7:8] / (X[:, 8:9] + 1e-6)
        intensity_variance = (X[:, 8:9] - X[:, 9:10]) / (X[:, 7:8] + 1e-6)
        intensity_ratio = X[:, 8:9] / (X[:, 9:10] + 1e-6)
        intensity_mean_diff = (X[:, 7:8] - (X[:, 8:9] + X[:, 9:10])/2)
        new_features.extend([intensity_range, intensity_contrast, intensity_normalized, 
                           intensity_variance, intensity_ratio, intensity_mean_diff])
        
        # === MORPHOLOGICAL FEATURES ===
        compactness = X[:, 0:1] / ((X[:, 1:2] + X[:, 2:3]) + 1e-6)
        circularity = 4 * np.pi * X[:, 0:1] / ((X[:, 1:2] * X[:, 2:3]) + 1e-6)
        shape_complexity = X[:, 4:5] * X[:, 3:4]
        elongation = np.maximum(X[:, 1:2], X[:, 2:3]) / (np.minimum(X[:, 1:2], X[:, 2:3]) + 1e-6)
        aspect_ratio = X[:, 1:2] / (X[:, 2:3] + 1e-6)
        solidity_extent_ratio = X[:, 4:5] / (X[:, 5:6] + 1e-6)
        axis_ratio_sq = (X[:, 1:2] / (X[:, 2:3] + 1e-6)) ** 2
        new_features.extend([compactness, circularity, shape_complexity, elongation,
                           aspect_ratio, solidity_extent_ratio, axis_ratio_sq])
        
        # === COMBINED FEATURES ===
        area_intensity = X[:, 0:1] * X[:, 7:8]
        perimeter_approx = 2 * (X[:, 1:2] + X[:, 2:3])
        area_perimeter_ratio = X[:, 0:1] / (perimeter_approx + 1e-6)
        eccentricity_intensity = X[:, 3:4] * X[:, 7:8]
        orientation_weighted = np.sin(X[:, 6:7]) * X[:, 0:1]
        solidity_area = X[:, 4:5] * X[:, 0:1]
        extent_intensity = X[:, 5:6] * X[:, 7:8]
        new_features.extend([area_intensity, area_perimeter_ratio, eccentricity_intensity,
                           orientation_weighted, solidity_area, extent_intensity])
        
        # === LOG/POWER TRANSFORMATIONS ===
        log_area = np.log10(X[:, 0:1] + 1)
        log_intensity = np.log10(X[:, 7:8] + 1)
        log_max_intensity = np.log10(X[:, 8:9] + 1)
        sqrt_area = np.sqrt(X[:, 0:1] + 1e-6)
        sqrt_intensity = np.sqrt(X[:, 7:8] + 1e-6)
        cube_root_area = np.cbrt(X[:, 0:1])
        new_features.extend([log_area, log_intensity, log_max_intensity, sqrt_area, 
                           sqrt_intensity, cube_root_area])
        
        # === RADIAL PROFILE (Enhanced) ===
        radial = X[:, 27:35]
        radial_gradients = np.diff(radial, axis=1)
        radial_mean = np.mean(radial, axis=1, keepdims=True)
        radial_std = np.std(radial, axis=1, keepdims=True)
        radial_max = np.max(radial, axis=1, keepdims=True)
        radial_min = np.min(radial, axis=1, keepdims=True)
        radial_max_min_ratio = radial_max / (radial_min + 1e-6)
        radial_skewness = skew(radial, axis=1, keepdims=True)
        radial_kurtosis = kurtosis(radial, axis=1, keepdims=True)
        radial_median = np.median(radial, axis=1, keepdims=True)
        radial_q1 = np.percentile(radial, 25, axis=1, keepdims=True)
        radial_q3 = np.percentile(radial, 75, axis=1, keepdims=True)
        radial_iqr = radial_q3 - radial_q1
        radial_cv = radial_std / (radial_mean + 1e-6)
        new_features.extend([radial_gradients, radial_mean, radial_std, radial_max_min_ratio,
                           radial_skewness, radial_kurtosis, radial_median, radial_iqr, radial_cv])
        
        # === HU MOMENTS (Enhanced) ===
        hu_moments = X[:, 10:17]
        hu_mean = np.mean(hu_moments, axis=1, keepdims=True)
        hu_std = np.std(hu_moments, axis=1, keepdims=True)
        hu_max = np.max(hu_moments, axis=1, keepdims=True)
        hu_min = np.min(hu_moments, axis=1, keepdims=True)
        hu_range = hu_max - hu_min
        hu_skewness = skew(hu_moments, axis=1, keepdims=True)
        hu_kurtosis = kurtosis(hu_moments, axis=1, keepdims=True)
        hu_median = np.median(hu_moments, axis=1, keepdims=True)
        new_features.extend([hu_mean, hu_std, hu_range, hu_skewness, hu_kurtosis, hu_median])
        
        # === LBP TEXTURE (Enhanced) ===
        lbp = X[:, 17:27]
        lbp_entropy = -np.sum(lbp * np.log(lbp + 1e-10), axis=1, keepdims=True)
        lbp_uniformity = np.sum(lbp ** 2, axis=1, keepdims=True)
        lbp_mean = np.mean(lbp, axis=1, keepdims=True)
        lbp_std = np.std(lbp, axis=1, keepdims=True)
        lbp_max = np.max(lbp, axis=1, keepdims=True)
        lbp_contrast = lbp_max - np.min(lbp, axis=1, keepdims=True)
        lbp_energy = np.sqrt(np.sum(lbp ** 2, axis=1, keepdims=True))
        lbp_skewness = skew(lbp, axis=1, keepdims=True)
        new_features.extend([lbp_entropy, lbp_uniformity, lbp_mean, lbp_std, 
                           lbp_contrast, lbp_energy, lbp_skewness])
        
        # === INTERACTION FEATURES ===
        area_eccentricity = X[:, 0:1] * X[:, 3:4]
        intensity_eccentricity = X[:, 7:8] * X[:, 3:4]
        solidity_circularity = X[:, 4:5] * circularity
        compactness_intensity = compactness * X[:, 7:8]
        new_features.extend([area_eccentricity, intensity_eccentricity, 
                           solidity_circularity, compactness_intensity])
        
        X_engineered = np.hstack([X] + new_features)
        
        # Handle NaN and Inf
        X_engineered = np.nan_to_num(X_engineered, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if verbose:
            print(f"  Original features: {X.shape[1]}")
            print(f"  Total features: {X_engineered.shape[1]}")
            print(f"  New features added: {X_engineered.shape[1] - X.shape[1]}")
        
        return X_engineered
    
    def split_and_oversample(self, X, y, use_smote):
        """Split data and apply SMOTE if requested"""
        print("\n" + "="*80)
        print("TRAIN/TEST SPLIT + OVERSAMPLING")
        print("="*80)
        
        # Split first
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"\nInitial split:")
        print(f"  Training: {self.X_train.shape[0]} samples")
        print(f"  Test: {self.X_test.shape[0]} samples")
        
        # Apply SMOTE to training set only
        if use_smote:
            print(f"\n🔬 Applying SMOTE oversampling to training set...")
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"  After SMOTE: {self.X_train.shape[0]} training samples")
            
            for i, class_name in enumerate(self.le.classes_):
                count = np.sum(self.y_train == i)
                print(f"    {class_name}: {count}")
    
    def train_ultimate_model(self, use_stacking=True):
        """Train ultimate model with stacking"""
        print("\n" + "="*80)
        print("TRAINING ULTIMATE MODEL")
        print("="*80)
        
        if use_stacking:
            print("\n🚀 Using Stacking Ensemble with Meta-Learner")
            
            # Base estimators
            estimators = [
                ('svm1', Pipeline([
                    ('scaler', RobustScaler()),
                    ('variance', VarianceThreshold(threshold=0.01)),
                    ('feature_selection', SelectKBest(mutual_info_classif, k=65)),
                    ('clf', SVC(kernel='rbf', C=15, gamma=0.04, class_weight='balanced',
                               probability=True, random_state=self.random_state))
                ])),
                ('svm2', Pipeline([
                    ('scaler', StandardScaler()),
                    ('variance', VarianceThreshold(threshold=0.01)),
                    ('feature_selection', SelectKBest(mutual_info_classif, k=60)),
                    ('clf', SVC(kernel='rbf', C=20, gamma=0.05, class_weight='balanced',
                               probability=True, random_state=self.random_state + 1))
                ])),
                ('svm3', Pipeline([
                    ('scaler', RobustScaler()),
                    ('variance', VarianceThreshold(threshold=0.01)),
                    ('feature_selection', SelectKBest(mutual_info_classif, k=55)),
                    ('clf', SVC(kernel='rbf', C=10, gamma=0.06, class_weight='balanced',
                               probability=True, random_state=self.random_state + 2))
                ])),
                ('rf', Pipeline([
                    ('scaler', StandardScaler()),
                    ('variance', VarianceThreshold(threshold=0.01)),
                    ('clf', RandomForestClassifier(n_estimators=100, max_depth=15, 
                                                   class_weight='balanced',
                                                   random_state=self.random_state))
                ]))
            ]
            
            # Meta-learner
            final_estimator = SVC(kernel='rbf', C=5, gamma=0.1, class_weight='balanced',
                                 probability=True, random_state=self.random_state)
            
            self.pipeline = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=5,
                n_jobs=-1
            )
            
        else:
            print("\n🔧 Using Single Optimized SVM")
            self.pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('variance', VarianceThreshold(threshold=0.01)),
                ('feature_selection', SelectKBest(mutual_info_classif, k=65)),
                ('svm', SVC(kernel='rbf', C=20, gamma=0.05, class_weight='balanced',
                           random_state=self.random_state))
            ])
        
        # Train
        print("\n⏳ Training ultimate model (this may take a while)...")
        start_time = time.time()
        self.pipeline.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        # Cross-validation on training set
        print("\n⏳ Performing 5-fold cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Use smaller subset for CV if training set is large
        if len(self.y_train) > 600:
            print("  (Using subset for faster CV)")
            cv_indices = np.random.choice(len(self.y_train), 600, replace=False)
            X_cv = self.X_train[cv_indices]
            y_cv = self.y_train[cv_indices]
        else:
            X_cv = self.X_train
            y_cv = self.y_train
            
        cv_scores = cross_val_score(self.pipeline, X_cv, y_cv, 
                                    cv=cv, scoring='f1_weighted', n_jobs=-1)
        
        print(f"\nCross-Validation Results:")
        print(f"  F1 Scores: {cv_scores}")
        print(f"  Mean F1: {cv_scores.mean():.4f}")
        print(f"  Std Dev: {cv_scores.std():.4f}")
        
        return cv_scores.mean(), cv_scores.std()
    
    def evaluate_test_set(self):
        """Evaluate on test set"""
        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)
        
        y_pred = self.pipeline.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        precision_weighted = precision_score(self.y_test, y_pred, average='weighted')
        recall_weighted = recall_score(self.y_test, y_pred, average='weighted')
        
        print(f"\n🏆 ULTIMATE TEST PERFORMANCE:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision_weighted:.4f}")
        print(f"  Recall:    {recall_weighted:.4f}")
        print(f"  F1 Score:  {f1_weighted:.4f}")
        
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.le.classes_, digits=4))
        
        cm = confusion_matrix(self.y_test, y_pred)
        print("\n📋 Confusion Matrix:")
        print(cm)
        
        print("\n📈 Per-Class Detailed Analysis:")
        for i, class_name in enumerate(self.le.classes_):
            class_mask = (self.y_test == i)
            class_total = np.sum(class_mask)
            class_correct = np.sum((self.y_test[class_mask] == y_pred[class_mask]))
            class_accuracy = class_correct / class_total if class_total > 0 else 0
            
            # Per-class precision and recall
            class_precision = precision_score(self.y_test, y_pred, labels=[i], average='macro')
            class_recall = recall_score(self.y_test, y_pred, labels=[i], average='macro')
            
            print(f"  {class_name}:")
            print(f"    Accuracy: {class_correct}/{class_total} ({class_accuracy:.1%})")
            print(f"    Precision: {class_precision:.4f}")
            print(f"    Recall: {class_recall:.4f}")
        
        return accuracy, f1_weighted, cm, y_pred
    
    def save_model(self, filename='ultimate_svm_model.joblib'):
        """Save model"""
        print(f"\n💾 Saving ultimate model to '{filename}'...")
        
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.le,
            'train_size': self.X_train.shape[0],
            'test_size': self.X_test.shape[0],
            'class_names': list(self.le.classes_),
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filename)
        print("✓ Ultimate model saved!")
    
    def plot_results(self, cm, accuracy, filename='ultimate_confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                    xticklabels=self.le.classes_,
                    yticklabels=self.le.classes_,
                    cbar_kws={'label': 'Count'})
        plt.title(f'🏆 Ultimate SVM Confusion Matrix\nAccuracy: {accuracy:.2%}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to '{filename}'")
        plt.close()

def main():
    print("="*80)
    print("🏆 ULTIMATE ASTRONOMICAL SVM TRAINING")
    print("="*80)
    print("\n✨ ULTIMATE Techniques Applied:")
    print("  1. Balanced undersampling (222 per class)")
    print("  2. SMOTE oversampling on training set")
    print("  3. Ultimate feature engineering (90+ features)")
    print("  4. Statistical features (skew, kurtosis, median, IQR)")
    print("  5. Stacking ensemble (3 SVMs + Random Forest + Meta-learner)")
    print("  6. RobustScaler for outlier resistance")
    print("  7. Variance threshold for noise removal")
    print("  8. Mutual Information feature selection")
    print("  9. Optimized hyperparameters")
    print()
    
    # Initialize trainer
    trainer = UltimateSVMTrainer(random_state=42)
    
    # Load and prepare
    X, y, use_smote = trainer.load_and_prepare_data(use_smote=True)
    
    # Engineer features
    X_eng = trainer.engineer_ultimate_features(X)
    
    # Split and oversample
    trainer.split_and_oversample(X_eng, y, use_smote)
    
    # Train ultimate model
    cv_mean, cv_std = trainer.train_ultimate_model(use_stacking=True)
    
    # Evaluate
    test_acc, test_f1, cm, y_pred = trainer.evaluate_test_set()
    
    # Save
    trainer.save_model('ultimate_svm_model_v2.joblib')
    trainer.plot_results(cm, test_acc, 'ultimate_confusion_matrix_v2.png')
    
    # Final summary
    print("\n" + "="*80)
    print("🏆 ULTIMATE TRAINING SUMMARY")
    print("="*80)
    print(f"\n✅ Cross-Validation: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"✅ Test F1 Score: {test_f1:.4f}")
    
    baseline = 0.5746
    improved = 0.6185
    print(f"\n📈 Performance Comparison:")
    print(f"  Baseline (train_model.py):    {baseline:.2%}")
    print(f"  Improved (train_model_improved.py): {improved:.2%} (+{(improved-baseline)*100:.2f}%)")
    print(f"  ULTIMATE (this):              {test_acc:.2%} (+{(test_acc-baseline)*100:.2f}%)")
    print(f"\n🎯 Total Improvement: {(test_acc-baseline)*100:+.2f} percentage points ({(test_acc/baseline-1)*100:+.1f}%)")
    
    print("\n" + "="*80)
    print("✅ ULTIMATE TRAINING COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
