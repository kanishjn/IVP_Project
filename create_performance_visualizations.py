#!/usr/bin/env python3
"""
Create comprehensive performance visualizations including per-class accuracy bar graphs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

def engineer_features(X):
    """Feature engineering matching ultimate_svm_model configuration"""
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
    radial_gradients = np.diff(radial, axis=1)
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
    
    return np.hstack([X] + new_features)

def get_per_class_metrics(y_true, y_pred, classes):
    """Calculate per-class accuracy and other metrics"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    metrics = {}
    for i, class_name in enumerate(classes):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        
        # Precision: of all predicted as this class, how many are correct
        predicted_as_class = cm[:, i].sum()
        precision = correct / predicted_as_class if predicted_as_class > 0 else 0
        
        # Recall is same as accuracy for this class
        recall = accuracy
        
        metrics[class_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'correct': correct,
            'total': total
        }
    
    return metrics

def create_per_class_accuracy_comparison(train_metrics, test_metrics, full_metrics, classes):
    """Create side-by-side bar graph comparing per-class accuracy across datasets"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))
    
    # Prepare data
    x = np.arange(len(classes))
    width = 0.25
    
    train_acc = [train_metrics[cls]['accuracy'] * 100 for cls in classes]
    test_acc = [test_metrics[cls]['accuracy'] * 100 for cls in classes]
    full_acc = [full_metrics[cls]['accuracy'] * 100 for cls in classes]
    
    # Create bars
    bars1 = plt.bar(x - width, train_acc, width, label='Training Set (532)', 
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = plt.bar(x, test_acc, width, label='Test Set (134)', 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = plt.bar(x + width, full_acc, width, label='Full Dataset (1000)', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    # Customize plot
    plt.xlabel('Class', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Per-Class Accuracy Comparison Across Datasets', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x, classes, fontsize=12)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=11, loc='lower right')
    plt.ylim(0, 100)
    
    # Add grid
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: per_class_accuracy_comparison.png")

def create_precision_recall_comparison(train_metrics, test_metrics, full_metrics, classes):
    """Create precision and recall comparison"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Precision comparison
    train_prec = [train_metrics[cls]['precision'] * 100 for cls in classes]
    test_prec = [test_metrics[cls]['precision'] * 100 for cls in classes]
    full_prec = [full_metrics[cls]['precision'] * 100 for cls in classes]
    
    bars1 = ax1.bar(x - width, train_prec, width, label='Training', 
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, test_prec, width, label='Test', 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax1.bar(x + width, full_prec, width, label='Full', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class Precision Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Recall comparison
    train_rec = [train_metrics[cls]['recall'] * 100 for cls in classes]
    test_rec = [test_metrics[cls]['recall'] * 100 for cls in classes]
    full_rec = [full_metrics[cls]['recall'] * 100 for cls in classes]
    
    bars1 = ax2.bar(x - width, train_rec, width, label='Training', 
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x, test_rec, width, label='Test', 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax2.bar(x + width, full_rec, width, label='Full', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class Recall Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('precision_recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: precision_recall_comparison.png")

def create_overall_accuracy_bar(train_acc, test_acc, full_acc):
    """Create overall accuracy comparison bar graph"""
    
    plt.figure(figsize=(10, 7))
    
    datasets = ['Training\n(532 samples)', 'Test\n(134 samples)', 'Full Dataset\n(1000 samples)']
    accuracies = [train_acc * 100, test_acc * 100, full_acc * 100]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = plt.bar(datasets, accuracies, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Overall Model Accuracy Across Datasets', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal line for reference
    plt.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='70% threshold')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('overall_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: overall_accuracy_comparison.png")

def main():
    print("="*80)
    print("CREATING PERFORMANCE VISUALIZATIONS")
    print("="*80)
    
    # Load model
    print("\n📦 Loading model...")
    model_data = joblib.load('improved_svm_model.joblib')
    pipeline = model_data['pipeline']
    le = model_data['label_encoder']
    print("✓ Model loaded")
    
    # Load data
    print("\n📂 Loading datasets...")
    X_full = np.load('features.npy')
    y_full = np.load('labels.npy')
    
    # Create balanced dataset and split
    balanced_indices = []
    samples_per_class = 222
    
    for class_name in ['GALAXY', 'QSO', 'STAR']:
        class_idx = np.where(y_full == class_name)[0]
        np.random.seed(42)
        selected = np.random.choice(class_idx, size=samples_per_class, replace=False)
        balanced_indices.extend(selected)
    
    balanced_indices = np.array(balanced_indices)
    np.random.seed(42)
    np.random.shuffle(balanced_indices)
    
    X_balanced = X_full[balanced_indices]
    y_balanced = y_full[balanced_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    print("✓ Data loaded and split")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Full: {len(X_full)} samples")
    
    # Engineer features and predict
    print("\n🔧 Engineering features and making predictions...")
    
    X_train_eng = engineer_features(X_train)
    X_test_eng = engineer_features(X_test)
    X_full_eng = engineer_features(X_full)
    
    y_train_pred = pipeline.predict(X_train_eng)
    y_test_pred = pipeline.predict(X_test_eng)
    y_full_pred = pipeline.predict(X_full_eng)
    
    y_train_pred_str = le.inverse_transform(y_train_pred)
    y_test_pred_str = le.inverse_transform(y_test_pred)
    y_full_pred_str = le.inverse_transform(y_full_pred)
    
    print("✓ Predictions complete")
    
    # Calculate metrics
    print("\n📊 Calculating per-class metrics...")
    classes = ['GALAXY', 'QSO', 'STAR']
    
    train_metrics = get_per_class_metrics(y_train, y_train_pred_str, classes)
    test_metrics = get_per_class_metrics(y_test, y_test_pred_str, classes)
    full_metrics = get_per_class_metrics(y_full, y_full_pred_str, classes)
    
    train_acc = accuracy_score(y_train, y_train_pred_str)
    test_acc = accuracy_score(y_test, y_test_pred_str)
    full_acc = accuracy_score(y_full, y_full_pred_str)
    
    # Print metrics
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY SUMMARY")
    print("="*80)
    
    print(f"\n{'Class':<10} {'Training':<15} {'Test':<15} {'Full Dataset'}")
    print("-" * 55)
    for cls in classes:
        train = train_metrics[cls]['accuracy'] * 100
        test = test_metrics[cls]['accuracy'] * 100
        full = full_metrics[cls]['accuracy'] * 100
        print(f"{cls:<10} {train:>6.2f}%  ({train_metrics[cls]['correct']}/{train_metrics[cls]['total']:<3})  "
              f"{test:>6.2f}%  ({test_metrics[cls]['correct']}/{test_metrics[cls]['total']:<3})  "
              f"{full:>6.2f}%  ({full_metrics[cls]['correct']}/{full_metrics[cls]['total']:<3})")
    
    print("\n" + "-" * 55)
    print(f"{'Overall':<10} {train_acc*100:>6.2f}%         {test_acc*100:>6.2f}%         {full_acc*100:>6.2f}%")
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    create_per_class_accuracy_comparison(train_metrics, test_metrics, full_metrics, classes)
    create_precision_recall_comparison(train_metrics, test_metrics, full_metrics, classes)
    create_overall_accuracy_bar(train_acc, test_acc, full_acc)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("="*80)
    
    print("\n📁 Generated Files:")
    print("  1. per_class_accuracy_comparison.png")
    print("  2. precision_recall_comparison.png")
    print("  3. overall_accuracy_comparison.png")
    
    print("\n🎯 Summary:")
    print(f"  Training Accuracy:  {train_acc*100:.2f}%")
    print(f"  Test Accuracy:      {test_acc*100:.2f}%")
    print(f"  Full Dataset:       {full_acc*100:.2f}%")
    print(f"  Overfitting Gap:    {(train_acc - test_acc)*100:+.2f}%")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
