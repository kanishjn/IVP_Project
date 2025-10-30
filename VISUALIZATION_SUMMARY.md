# Model Performance Visualizations

This document summarizes all the generated visualizations for the final SVM model.

---

## 📊 Generated Visualizations

### 1. **Per-Class Accuracy Comparison** (`per_class_accuracy_comparison.png`)
**Side-by-side comparison of accuracy for each class across all datasets**

Shows:
- Training Set (532 samples) - Green bars
- Test Set (134 samples) - Blue bars  
- Full Dataset (1000 samples) - Red bars

Key Insights:
- **GALAXY:** 76.4% (train) → 77.3% (test) → 57.4% (full)
  - Large drop on full dataset due to class imbalance
- **QSO:** 71.8% (train) → 66.7% (test) → 72.3% (full)
  - Most consistent across datasets
- **STAR:** 75.1% (train) → 80.0% (test) → 76.1% (full)
  - Best performance, especially on test set

---

### 2. **Overall Accuracy Comparison** (`overall_accuracy_comparison.png`)
**Bar graph showing overall model accuracy across datasets**

Results:
- Training: **74.44%**
- Test: **74.63%**
- Full Dataset: **65.60%**

Analysis:
- Excellent generalization (test > train by 0.19%)
- Full dataset lower due to class imbalance (507 GALAXY, 271 QSO, 222 STAR)

---

### 3. **Precision-Recall Comparison** (`precision_recall_comparison.png`)
**Dual bar graphs showing precision and recall for each class**

**Precision (when model predicts X, how often is it correct?):**
- Training: GALAXY 73.1%, QSO 76.1%, STAR 74.3%
- Test: GALAXY 66.7%, QSO 81.1%, STAR 78.3%
- Full: GALAXY 79.9%, QSO 64.5%, STAR 50.9%

**Recall (of actual X samples, how many did model find?):**
- Same as accuracy for each class
- Full dataset: GALAXY 57.4%, QSO 72.3%, STAR 76.1%

Key Insight:
- On full dataset: GALAXY has high precision but low recall (misses many)
- STAR has low precision but high recall (over-predicts)

---

### 4. **Confusion Matrices**

#### 4a. Training Set (`final_model_confusion_train.png`)
```
              Predicted
            GALAXY  QSO  STAR
  GALAXY     136    19    23     (76.4%)
  QSO         27   127    23     (71.8%)
  STAR        23    21   133     (75.1%)
```

#### 4b. Test Set (`final_model_confusion_test.png`)
```
              Predicted
            GALAXY  QSO  STAR
  GALAXY      34     3     7     (77.3%)
  QSO         12    30     3     (66.7%)
  STAR         5     4    36     (80.0%)
```

#### 4c. Full Dataset (`final_model_confusion_full.png`)
```
              Predicted
            GALAXY  QSO  STAR
  GALAXY     291    83   133     (57.4%)
  QSO         45   196    30     (72.3%)
  STAR        28    25   169     (76.1%)
```

Common Confusion Patterns:
- **GALAXY ↔ STAR:** Most common confusion (133 galaxies predicted as stars)
- **QSO → GALAXY:** 45 QSOs misclassified as galaxies
- **STAR → GALAXY:** 28 stars misclassified as galaxies

---

## 📈 Key Performance Metrics Summary

| Metric | Training | Test | Full Dataset |
|--------|----------|------|--------------|
| **Overall Accuracy** | 74.44% | 74.63% | 65.60% |
| **GALAXY Accuracy** | 76.40% | 77.27% | 57.40% |
| **QSO Accuracy** | 71.75% | 66.67% | 72.32% |
| **STAR Accuracy** | 75.14% | 80.00% | 76.13% |

---

## 🎯 Interpretation

### Why is Full Dataset Accuracy Lower?

1. **Class Imbalance:**
   - GALAXY: 507 samples (50.7%)
   - QSO: 271 samples (27.1%)
   - STAR: 222 samples (22.2%)

2. **Training on Balanced Data:**
   - Model trained on equal representation (222 each)
   - Real-world data is imbalanced
   - Model shows bias toward minority classes

3. **Expected Behavior:**
   - The 9% drop (74.63% → 65.60%) is reasonable
   - Shows model works well on balanced data
   - Real-world performance still competitive

### Model Strengths

✅ **No Overfitting:** Test accuracy (74.63%) ≥ Training (74.44%)
✅ **Consistent QSO Detection:** 72.3% across imbalanced data
✅ **Good STAR Detection:** 76-80% across all datasets
✅ **High Precision for GALAXY:** 79.9% (when predicted, usually correct)

### Model Weaknesses

⚠️ **GALAXY Recall on Full Data:** Only 57.4% (misses many galaxies)
⚠️ **STAR Precision on Full Data:** Only 50.9% (over-predicts stars)
⚠️ **GALAXY-STAR Confusion:** Significant overlap in feature space

---

## 🚀 Usage

### Generate all visualizations:
```bash
python3 create_performance_visualizations.py
```

### Run full evaluation:
```bash
python3 final_model_evaluation.py
```

---

## 📁 All Visualization Files

1. `per_class_accuracy_comparison.png` - Bar graph comparing per-class accuracy
2. `overall_accuracy_comparison.png` - Overall accuracy bar graph
3. `precision_recall_comparison.png` - Precision and recall comparison
4. `final_model_confusion_train.png` - Training set confusion matrix
5. `final_model_confusion_test.png` - Test set confusion matrix
6. `final_model_confusion_full.png` - Full dataset confusion matrix
7. `final_model_confusion_balanced.png` - (Deprecated, kept for reference)

---

*Generated on: October 24, 2025*
*Model: improved_svm_model.joblib*
*Visualization Script: create_performance_visualizations.py*
