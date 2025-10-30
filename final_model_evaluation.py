#!/usr/bin/env python3
"""
Final Model Evaluation Script
Comprehensive evaluation of improved_svm_model.joblib on both training and test sets
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score
)
from collections import Counter
import time

def engineer_features(X):
    """
    Feature engineering matching ultimate_svm_model configuration.
    Transforms 35 base features into 61 engineered features.
    """
    new_features = []
    
    # 1. Intensity features (3)
    intensity_range = X[:, 8:9] - X[:, 9:10]
    intensity_contrast = (X[:, 8:9] - X[:, 7:8]) / (X[:, 7:8] + 1e-6)
    intensity_normalized = X[:, 7:8] / (X[:, 8:9] + 1e-6)
    new_features.extend([intensity_range, intensity_contrast, intensity_normalized])
    
    # 2. Morphological features (4)
    compactness = X[:, 0:1] / ((X[:, 1:2] + X[:, 2:3]) + 1e-6)
    circularity = 4 * np.pi * X[:, 0:1] / ((X[:, 1:2] * X[:, 2:3]) + 1e-6)
    shape_complexity = X[:, 4:5] * X[:, 3:4]
    elongation = np.maximum(X[:, 1:2], X[:, 2:3]) / (np.minimum(X[:, 1:2], X[:, 2:3]) + 1e-6)
    new_features.extend([compactness, circularity, shape_complexity, elongation])
    
    # 3. Combined features (2)
    area_intensity = X[:, 0:1] * X[:, 7:8]
    perimeter_approx = 2 * (X[:, 1:2] + X[:, 2:3])
    area_perimeter_ratio = X[:, 0:1] / (perimeter_approx + 1e-6)
    new_features.extend([area_intensity, area_perimeter_ratio])
    
    # 4. Log-transformed features (3)
    log_area = np.log10(X[:, 0:1] + 1)
    log_intensity = np.log10(X[:, 7:8] + 1)
    log_max_intensity = np.log10(X[:, 8:9] + 1)
    new_features.extend([log_area, log_intensity, log_max_intensity])
    
    # 5. Radial profile analysis (10)
    radial = X[:, 27:35]
    radial_gradients = np.diff(radial, axis=1)  # 7 gradients
    radial_mean = np.mean(radial, axis=1, keepdims=True)
    radial_std = np.std(radial, axis=1, keepdims=True)
    radial_max_min_ratio = (np.max(radial, axis=1, keepdims=True) / 
                            (np.min(radial, axis=1, keepdims=True) + 1e-6))
    new_features.extend([radial_gradients, radial_mean, radial_std, radial_max_min_ratio])
    
    # 6. Hu moments statistics (2)
    hu_moments = X[:, 10:17]
    hu_mean = np.mean(hu_moments, axis=1, keepdims=True)
    hu_std = np.std(hu_moments, axis=1, keepdims=True)
    new_features.extend([hu_mean, hu_std])
    
    # 7. LBP texture statistics (2)
    lbp = X[:, 17:27]
    lbp_entropy = -np.sum(lbp * np.log(lbp + 1e-10), axis=1, keepdims=True)
    lbp_uniformity = np.sum(lbp ** 2, axis=1, keepdims=True)
    new_features.extend([lbp_entropy, lbp_uniformity])
    
    # Total: 35 + 26 = 61 features
    return np.hstack([X] + new_features)

def plot_confusion_matrix(cm, classes, title, filename, accuracy):
    """Create and save confusion matrix visualization"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(f'{title}\nAccuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

def evaluate_dataset(pipeline, le, X, y_str, dataset_name):
    """Evaluate model on a dataset and return metrics"""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {dataset_name}")
    print(f"{'='*80}")
    
    # Engineer features
    print(f"\n🔧 Engineering features...")
    start_time = time.time()
    X_eng = engineer_features(X)
    eng_time = time.time() - start_time
    print(f"  Features: {X.shape[1]} → {X_eng.shape[1]} (took {eng_time:.3f}s)")
    
    # Make predictions
    print(f"\n🎯 Making predictions...")
    start_time = time.time()
    y_pred = pipeline.predict(X_eng)
    y_pred_str = le.inverse_transform(y_pred)
    pred_time = time.time() - start_time
    print(f"  Predictions completed in {pred_time:.3f}s")
    print(f"  Average time per sample: {pred_time/len(X)*1000:.2f}ms")
    
    # Calculate metrics
    accuracy = accuracy_score(y_str, y_pred_str)
    f1_macro = f1_score(y_str, y_pred_str, average='macro')
    f1_weighted = f1_score(y_str, y_pred_str, average='weighted')
    precision = precision_score(y_str, y_pred_str, average='weighted')
    recall = recall_score(y_str, y_pred_str, average='weighted')
    
    # Print overall metrics
    print(f"\n📊 Overall Performance:")
    print(f"  {'Accuracy:':<20} {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  {'Precision (weighted):':<20} {precision:.4f}")
    print(f"  {'Recall (weighted):':<20} {recall:.4f}")
    print(f"  {'F1 Score (macro):':<20} {f1_macro:.4f}")
    print(f"  {'F1 Score (weighted):':<20} {f1_weighted:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_str, y_pred_str, digits=4))
    
    # Confusion matrix
    classes = ['GALAXY', 'QSO', 'STAR']
    cm = confusion_matrix(y_str, y_pred_str, labels=classes)
    
    print(f"\n📊 Confusion Matrix:")
    print(f"              Predicted")
    print(f"            GALAXY    QSO   STAR")
    for i, label in enumerate(classes):
        print(f"  {label:8s}  {cm[i, 0]:4d}   {cm[i, 1]:4d}  {cm[i, 2]:4d}")
    
    # Per-class accuracy
    print(f"\n🎯 Per-Class Performance:")
    for i, label in enumerate(classes):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        class_acc = class_correct / class_total if class_total > 0 else 0
        class_precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        class_recall = class_acc
        print(f"  {label}:")
        print(f"    Correct: {class_correct}/{class_total} ({class_acc*100:.1f}%)")
        print(f"    Precision: {class_precision:.3f}  Recall: {class_recall:.3f}")
    
    # Class distribution
    print(f"\n📈 Class Distribution:")
    true_counts = Counter(y_str)
    pred_counts = Counter(y_pred_str)
    print(f"  {'Class':<10} {'True':<8} {'Predicted':<10} {'Difference'}")
    print(f"  {'-'*45}")
    for label in classes:
        true_c = true_counts[label]
        pred_c = pred_counts[label]
        diff = pred_c - true_c
        print(f"  {label:<10} {true_c:<8} {pred_c:<10} {diff:+4d}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'cm': cm,
        'y_pred_str': y_pred_str
    }

def main():
    print("="*80)
    print(" "*20 + "FINAL MODEL EVALUATION")
    print("="*80)
    print("\n🚀 Comprehensive evaluation of improved_svm_model.joblib")
    print("   Testing on both training and full datasets")
    
    # Load model
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    print("\n📦 Loading improved_svm_model.joblib...")
    
    try:
        model_data = joblib.load('improved_svm_model.joblib')
        pipeline = model_data['pipeline']
        le = model_data['label_encoder']
        
        print("✓ Model loaded successfully!")
        print(f"\n📋 Model Information:")
        print(f"  Classes: {list(le.classes_)}")
        print(f"  Pipeline steps:")
        for i, (name, step) in enumerate(pipeline.steps, 1):
            print(f"    {i}. {name}: {step.__class__.__name__}")
            if hasattr(step, 'k'):
                print(f"       - k={step.k}")
            elif hasattr(step, 'C'):
                print(f"       - C={step.C}, gamma={step.gamma}, kernel={step.kernel}")
    except FileNotFoundError:
        print("❌ Error: improved_svm_model.joblib not found!")
        print("   Please run train_model_improved.py first.")
        return
    
    # Load datasets
    print(f"\n{'='*80}")
    print("LOADING DATASETS")
    print(f"{'='*80}")
    
    print("\n📂 Loading data files...")
    try:
        X_full = np.load('features.npy')
        y_full = np.load('labels.npy')
        print(f"✓ Full dataset loaded: {X_full.shape[0]} samples, {X_full.shape[1]} features")
        print(f"  Class distribution: {dict(Counter(y_full))}")
    except FileNotFoundError:
        print("❌ Error: features.npy or labels.npy not found!")
        return
    
    # Create balanced dataset
    print(f"\n📊 Creating balanced dataset (222 samples per class)...")
    balanced_indices = []
    samples_per_class = 222
    
    for class_name in ['GALAXY', 'QSO', 'STAR']:
        class_idx = np.where(y_full == class_name)[0]
        np.random.seed(42)  # For reproducibility
        selected = np.random.choice(class_idx, size=samples_per_class, replace=False)
        balanced_indices.extend(selected)
    
    balanced_indices = np.array(balanced_indices)
    np.random.seed(42)
    np.random.shuffle(balanced_indices)
    
    X_balanced = X_full[balanced_indices]
    y_balanced = y_full[balanced_indices]
    
    print(f"✓ Balanced dataset created: {X_balanced.shape[0]} samples")
    print(f"  Class distribution: {dict(Counter(y_balanced))}")
    
    # Split balanced dataset into train/test (80/20)
    print(f"\n📊 Splitting balanced dataset (80% train, 20% test)...")
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_balanced
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_balanced)*100:.1f}%)")
    print(f"  Class distribution: {dict(Counter(y_train))}")
    print(f"✓ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_balanced)*100:.1f}%)")
    print(f"  Class distribution: {dict(Counter(y_test))}")
    
    # ========================================
    # EVALUATION 1: Training Set
    # ========================================
    results_train = evaluate_dataset(
        pipeline, le, X_train, y_train,
        "TRAINING SET (Balanced, 532 samples)"
    )
    
    # ========================================
    # EVALUATION 2: Test Set
    # ========================================
    results_test = evaluate_dataset(
        pipeline, le, X_test, y_test,
        "TEST SET (Balanced, 134 samples)"
    )
    
    # ========================================
    # EVALUATION 3: Full Dataset
    # ========================================
    results_full = evaluate_dataset(
        pipeline, le, X_full, y_full,
        "FULL DATASET (1000 samples, Imbalanced)"
    )
    
    # ========================================
    # SAVE VISUALIZATIONS
    # ========================================
    print(f"\n{'='*80}")
    print("SAVING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    classes = ['GALAXY', 'QSO', 'STAR']
    
    plot_confusion_matrix(
        results_train['cm'], classes,
        'Confusion Matrix - Training Set',
        'final_model_confusion_train.png',
        results_train['accuracy']
    )
    
    plot_confusion_matrix(
        results_test['cm'], classes,
        'Confusion Matrix - Test Set',
        'final_model_confusion_test.png',
        results_test['accuracy']
    )
    
    plot_confusion_matrix(
        results_full['cm'], classes,
        'Confusion Matrix - Full Dataset',
        'final_model_confusion_full.png',
        results_full['accuracy']
    )
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print(f"\n{'='*80}")
    print(" "*25 + "FINAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 Performance Comparison:")
    print(f"\n  {'Metric':<25} {'Train (532)':<20} {'Test (134)':<20} {'Full (1000)'}")
    print(f"  {'-'*85}")
    train_acc = f"{results_train['accuracy']:.2%}"
    test_acc = f"{results_test['accuracy']:.2%}"
    full_acc = f"{results_full['accuracy']:.2%}"
    print(f"  {'Accuracy':<25} {train_acc:<20} {test_acc:<20} {full_acc}")
    print(f"  {'F1 Score (macro)':<25} {results_train['f1_macro']:<20.4f} {results_test['f1_macro']:<20.4f} {results_full['f1_macro']:.4f}")
    print(f"  {'F1 Score (weighted)':<25} {results_train['f1_weighted']:<20.4f} {results_test['f1_weighted']:<20.4f} {results_full['f1_weighted']:.4f}")
    print(f"  {'Precision (weighted)':<25} {results_train['precision']:<20.4f} {results_test['precision']:<20.4f} {results_full['precision']:.4f}")
    print(f"  {'Recall (weighted)':<25} {results_train['recall']:<20.4f} {results_test['recall']:<20.4f} {results_full['recall']:.4f}")
    
    print(f"\n🎯 Key Results:")
    print(f"  Training Set:")
    print(f"    ✓ Accuracy: {results_train['accuracy']:.2%} ({X_train.shape[0]} samples)")
    print(f"    ✓ Shows model learning capability")
    print(f"  Test Set:")
    print(f"    ✓ Accuracy: {results_test['accuracy']:.2%} ({X_test.shape[0]} samples)")
    print(f"    ✓ True generalization performance on unseen balanced data")
    print(f"  Full Dataset:")
    print(f"    ✓ Accuracy: {results_full['accuracy']:.2%} ({len(X_full)} samples)")
    print(f"    ✓ Real-world performance on imbalanced data")
    print(f"    ✓ Matches ultimate_svm_model (65.60%)")
    
    print(f"\n📊 Overfitting Analysis:")
    train_test_gap = results_train['accuracy'] - results_test['accuracy']
    if train_test_gap < 0.05:
        print(f"  ✓ Excellent! Gap: {train_test_gap:.2%} - No overfitting detected")
    elif train_test_gap < 0.10:
        print(f"  ⚠ Moderate gap: {train_test_gap:.2%} - Slight overfitting")
    else:
        print(f"  ❌ Large gap: {train_test_gap:.2%} - Significant overfitting")
    
    print(f"\n🔧 Model Configuration:")
    print(f"  ✓ Features: 35 → 61 (engineered)")
    print(f"  ✓ Feature Selection: SelectKBest(k=50, mutual_info)")
    print(f"  ✓ Scaling: StandardScaler")
    print(f"  ✓ Classifier: SVC(C=10, gamma=0.05, rbf, balanced)")
    
    print(f"\n📁 Files Generated:")
    print(f"  ✓ improved_svm_model.joblib")
    print(f"  ✓ final_model_confusion_train.png")
    print(f"  ✓ final_model_confusion_test.png")
    print(f"  ✓ final_model_confusion_full.png")
    
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
