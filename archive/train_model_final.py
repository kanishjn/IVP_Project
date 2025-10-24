#!/usr/bin/env python3
"""
FINAL OPTIMIZED SVM Training - With Grid Search
Finds the absolute best hyperparameters for maximum accuracy
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            f1_score, make_scorer)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from collections import Counter
from scipy.stats import skew, kurtosis
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Import from train_model_improved
import sys
sys.path.append('.')
from train_model_improved import ImprovedSVMTrainer

def main():
    print("="*80)
    print("🔬 FINAL OPTIMIZED SVM TRAINING - GRID SEARCH")
    print("="*80)
    print("\nThis will find the BEST possible hyperparameters...")
    print("Expected runtime: 5-10 minutes\n")
    
    # Initialize
    trainer = ImprovedSVMTrainer(random_state=42)
    
    # Load and prepare
    print("Loading and preparing data...")
    X, y = trainer.load_and_prepare_data(use_augmentation=True)
    
    # Engineer features
    X_eng = trainer.engineer_advanced_features(X, verbose=True)
    
    # Split
    trainer.split_data(X_eng, y)
    
    # Grid Search Optimization
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE GRID SEARCH")
    print("="*80)
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('variance', VarianceThreshold(threshold=0.01)),
        ('feature_selection', SelectKBest(mutual_info_classif)),
        ('svm', SVC(class_weight='balanced', random_state=42))
    ])
    
    param_grid = {
        'feature_selection__k': [50, 55, 60, 65, 70, 75],
        'svm__C': [5, 10, 15, 20, 25, 30],
        'svm__gamma': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        'svm__kernel': ['rbf']
    }
    
    total_combinations = (len(param_grid['feature_selection__k']) * 
                         len(param_grid['svm__C']) * 
                         len(param_grid['svm__gamma']))
    
    print(f"\nParameter Space:")
    print(f"  Features: {param_grid['feature_selection__k']}")
    print(f"  C values: {param_grid['svm__C']}")
    print(f"  Gamma values: {param_grid['svm__gamma']}")
    print(f"\nTotal combinations: {total_combinations}")
    print(f"With 5-fold CV: {total_combinations * 5} fits")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    print(f"\n⏳ Starting grid search...")
    start_time = time.time()
    
    grid_search.fit(trainer.X_train, trainer.y_train)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Grid search completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    
    # Best parameters
    print("\n" + "="*80)
    print("🏆 BEST PARAMETERS FOUND")
    print("="*80)
    
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\n✅ Best CV F1 Score: {grid_search.best_score_:.4f}")
    
    # Show top 10 configurations
    print("\n" + "="*80)
    print("📊 TOP 10 CONFIGURATIONS")
    print("="*80)
    
    results = grid_search.cv_results_
    indices = np.argsort(results['mean_test_score'])[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        score = results['mean_test_score'][idx]
        std = results['std_test_score'][idx]
        params = results['params'][idx]
        
        print(f"\n#{i} - F1: {score:.4f} (±{std:.4f})")
        print(f"    k={params['feature_selection__k']}, C={params['svm__C']}, gamma={params['svm__gamma']}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("🎯 TEST SET EVALUATION")
    print("="*80)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(trainer.X_test)
    
    accuracy = accuracy_score(trainer.y_test, y_pred)
    f1_weighted = f1_score(trainer.y_test, y_pred, average='weighted')
    
    print(f"\n🏆 Test Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score: {f1_weighted:.4f}")
    
    print("\n📊 Classification Report:")
    print(classification_report(trainer.y_test, y_pred, target_names=trainer.le.classes_, digits=4))
    
    cm = confusion_matrix(trainer.y_test, y_pred)
    print("\n📋 Confusion Matrix:")
    print(cm)
    
    print("\n📈 Per-Class Analysis:")
    for i, class_name in enumerate(trainer.le.classes_):
        class_mask = (trainer.y_test == i)
        class_total = np.sum(class_mask)
        class_correct = np.sum((trainer.y_test[class_mask] == y_pred[class_mask]))
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"  {class_name}: {class_correct}/{class_total} ({class_accuracy:.1%})")
    
    # Save model
    print(f"\n💾 Saving optimized model...")
    
    model_data = {
        'pipeline': best_model,
        'label_encoder': trainer.le,
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': accuracy,
        'test_f1': f1_weighted,
        'train_size': trainer.X_train.shape[0],
        'test_size': trainer.X_test.shape[0],
        'grid_search_results': grid_search.cv_results_
    }
    
    joblib.dump(model_data, 'final_optimized_svm.joblib')
    print("✓ Model saved to 'final_optimized_svm.joblib'")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=trainer.le.classes_,
                yticklabels=trainer.le.classes_,
                cbar_kws={'label': 'Count'})
    plt.title(f'Final Optimized SVM\nAccuracy: {accuracy:.2%}', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig('final_optimized_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to 'final_optimized_confusion_matrix.png'")
    plt.close()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON")
    print("="*80)
    
    baseline = 0.5746
    print(f"\nPerformance Progression:")
    print(f"  Baseline (train_model.py):        {baseline:.2%}")
    print(f"  Improved (train_model_improved):  ~{0.5954:.2%} (+{(0.5954-baseline)*100:.2f}%)")
    print(f"  FINAL OPTIMIZED (this):           {accuracy:.2%} (+{(accuracy-baseline)*100:.2f}%)")
    
    improvement = (accuracy - baseline) * 100
    relative = (accuracy / baseline - 1) * 100
    
    print(f"\n🎯 Total Improvement:")
    print(f"  Absolute: +{improvement:.2f} percentage points")
    print(f"  Relative: +{relative:.1f}%")
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"\nBest configuration:")
    print(f"  Features: {grid_search.best_params_['feature_selection__k']}")
    print(f"  C: {grid_search.best_params_['svm__C']}")
    print(f"  Gamma: {grid_search.best_params_['svm__gamma']}")

if __name__ == '__main__':
    main()
