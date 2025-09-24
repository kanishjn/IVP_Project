# train_eval_adaptive.py
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Load features & labels
X = np.load("features.npy")
y = np.load("labels.npy")

# Encode labels -> integers
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", le.classes_)

# Show class distribution
unique, counts = np.unique(y_enc, return_counts=True)
class_counts = dict(zip(le.classes_, counts))
print("Class counts:", class_counts)
min_count = counts.min()

# Decide splitting strategy
# If min_count >= 2, we can do stratified split with test_size 0.5 (ensures 1 per class in test)
# But if dataset is tiny, prefer StratifiedKFold for cross-validation (n_splits = min_count)
if len(X) < 20:
    # small dataset — use StratifiedKFold for evaluation
    n_splits = max(2, min(5, min_count))  # at least 2
    print(f"Small dataset detected ({len(X)} samples). Using StratifiedKFold with n_splits={n_splits}.")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, class_weight="balanced", random_state=42))
    ])
    param_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.01],
        "svc__kernel": ["rbf"]
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=2)
    grid.fit(X, y_enc)
    print("Best params:", grid.best_params_)
    best = grid.best_estimator_

    # cross-validated scores
    scores = cross_val_score(best, X, y_enc, cv=cv, scoring="f1_weighted")
    print(f"Cross-validated f1_weighted scores: {scores}")
    print(f"Mean CV f1_weighted: {scores.mean():.3f} ± {scores.std():.3f}")

    # For a final report, do a single split if possible (train_test_split) to show confusion matrix:
    if min_count >= 1 and len(X) >= len(le.classes_):
        test_size = 0.5 if len(X) < 20 else 0.2
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=test_size, stratify=y_enc, random_state=42
            )
            best.fit(X_train, y_train)
            y_pred = best.predict(X_test)
            print("Final evaluation on a held-out split:")
            print(classification_report(y_test, y_pred, target_names=le.classes_))
            print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        except Exception as e:
            warnings.warn(f"Could not run held-out split: {e}")
    # Save model
    joblib.dump({"pipeline": best, "label_encoder": le}, "svm_pipeline_adaptive.joblib")
    print("Saved model to svm_pipeline_adaptive.joblib")

else:
    # larger dataset: standard train/test + GridSearchCV
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, stratify=y_enc, random_state=42
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, class_weight="balanced", random_state=42))
    ])
    param_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.01, 0.001],
        "svc__kernel": ["rbf"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    joblib.dump({"pipeline": best, "label_encoder": le}, "svm_pipeline.joblib")
    print("Saved model to svm_pipeline.joblib")