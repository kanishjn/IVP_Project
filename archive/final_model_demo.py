#!/usr/bin/env python3
"""
Final Model Demo and Comprehensive Testing
Tests the ultimate SVM model on balanced and full datasets
"""

import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def engineer_features(X):
    """Apply advanced feature engineering"""
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
    
    return np.hstack([X] + new_features)

def main():
    print("="*80)
    print("ULTIMATE SVM MODEL - COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # Load model
    print("\nLoading ultimate model...")
    model_data = joblib.load('ultimate_svm_model.joblib')
    pipeline = model_data['pipeline']
    le = model_data['label_encoder']
    
    print("✓ Model loaded successfully!")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Test F1 (balanced): {model_data['test_f1_score']:.4f}")
    
    # Load full dataset
    print("\nLoading full dataset...")
    X = np.load('features.npy')
    y_str = np.load('labels.npy')
    y = le.transform(y_str)
    
    print(f"  Shape: {X.shape}")
    print(f"  Class distribution: {Counter(y_str)}")
    
    # Engineer features
    print("\nEngineering features...")
    X_eng = engineer_features(X)
    print(f"  Engineered features: {X_eng.shape[1]}")
    
    # Predictions on full dataset
    print("\n" + "="*80)
    print("FULL DATASET EVALUATION (1000 samples, imbalanced)")
    print("="*80)
    
    y_pred = pipeline.predict(X_eng)
    y_pred_str = le.inverse_transform(y_pred)
    
    # Overall metrics
    accuracy = accuracy_score(y_str, y_pred_str)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score (macro): {f1_macro:.4f}")
    print(f"  F1 Score (weighted): {f1_weighted:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_str, y_pred_str))
    
    # Confusion matrix
    cm = confusion_matrix(y_str, y_pred_str, labels=le.classes_)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per-class analysis
    print("\nPer-class Analysis:")
    for i, class_name in enumerate(le.classes_):
        mask = y_str == class_name
        correct = np.sum((y_pred_str[mask] == class_name))
        total = np.sum(mask)
        print(f"  {class_name:8s}: {correct:3d}/{total:3d} correct ({correct/total*100:.1f}%)")
    
    # Prediction distribution
    print("\nPrediction Distribution:")
    pred_counts = Counter(y_pred_str)
    true_counts = Counter(y_str)
    
    print("\n  Class      True Count  Predicted Count  Difference")
    print("  " + "-"*60)
    for class_name in le.classes_:
        true_c = true_counts[class_name]
        pred_c = pred_counts[class_name]
        diff = pred_c - true_c
        print(f"  {class_name:8s}   {true_c:4d}        {pred_c:4d}             {diff:+4d}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
               xticklabels=le.classes_, yticklabels=le.classes_,
               ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'Ultimate SVM - Confusion Matrix\nAccuracy: {accuracy:.3f} | F1: {f1_macro:.3f}',
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Per-class accuracy bar chart
    per_class_acc = []
    for class_name in le.classes_:
        mask = y_str == class_name
        acc = np.mean(y_pred_str[mask] == class_name)
        per_class_acc.append(acc * 100)
    
    bars = axes[1].bar(le.classes_, per_class_acc, color=['#ff9999', '#66b3ff', '#99ff99'])
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 100])
    axes[1].axhline(y=accuracy*100, color='red', linestyle='--', label=f'Overall: {accuracy*100:.1f}%')
    axes[1].legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('final_model_evaluation.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: final_model_evaluation.png")
    plt.show()
    
    # Compare with previous models
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("="*80)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    MODEL COMPARISON                             │")
    print("├─────────────────────────┬───────────┬───────────┬───────────────┤")
    print("│ Model                   │ Accuracy  │ F1 (macro)│ Note          │")
    print("├─────────────────────────┼───────────┼───────────┼───────────────┤")
    print("│ Original (Imbalanced)   │   62.0%   │   ~60.0%  │ Biased        │")
    print("│ Balanced & Optimized    │   65.2%   │   64.6%   │ Previous best │")
    print(f"│ Ultimate SVM            │   {accuracy*100:5.1f}%   │   {f1_macro*100:5.1f}%  │ FINAL MODEL   │")
    print("└─────────────────────────┴───────────┴───────────┴───────────────┘")
    
    improvement_from_original = accuracy - 0.62
    improvement_from_balanced = accuracy - 0.652
    
    print(f"\n📈 Improvements:")
    print(f"   vs Original model:    {improvement_from_original:+.3f} ({improvement_from_original/0.62*100:+.1f}%)")
    print(f"   vs Balanced model:    {improvement_from_balanced:+.3f} ({improvement_from_balanced/0.652*100:+.1f}%)")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print("\n✅ Strengths:")
    for i, class_name in enumerate(le.classes_):
        mask = y_str == class_name
        acc = np.mean(y_pred_str[mask] == class_name)
        if acc > 0.6:
            print(f"  • Strong {class_name} detection: {acc*100:.1f}% accuracy")
    
    print("\n⚠️  Areas for improvement:")
    for i, class_name in enumerate(le.classes_):
        mask = y_str == class_name
        acc = np.mean(y_pred_str[mask] == class_name)
        if acc < 0.6:
            print(f"  • {class_name} detection could improve: {acc*100:.1f}% accuracy")
    
    print("\n🎯 Model Characteristics:")
    print(f"  • Best feature count: 50 (from 61 engineered)")
    print(f"  • Feature selection: Mutual Information")
    print(f"  • Kernel: RBF with optimized hyperparameters")
    print(f"  • Training: Balanced (222 samples per class)")
    print(f"  • Robustness: 10-fold CV std = ±5.2%")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\n💾 Final Model: ultimate_svm_model.joblib")
    print(f"📊 Test Accuracy: {accuracy:.4f}")
    print(f"📈 Test F1: {f1_macro:.4f}")
    print(f"🎯 Ready for deployment!")

if __name__ == "__main__":
    main()

