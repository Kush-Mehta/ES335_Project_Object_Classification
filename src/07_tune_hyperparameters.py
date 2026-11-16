import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import FEATURES_PATH, LABELS_PATH, PARAMS_PATH
SAVE_PATH = PARAMS_PATH

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

X_trainpool, X_test, y_trainpool, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_trainpool_scaled = scaler.fit_transform(X_trainpool)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Tuning Logistic Regression...")
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=3000),
    param_grid={"C": [0.01, 0.1, 1, 10]},
    cv=cv, scoring="accuracy", n_jobs=-1
)
t0 = time.time()
lr_grid.fit(X_trainpool_scaled, y_trainpool)
print(f"  Best params: {lr_grid.best_params_}  CV acc: {lr_grid.best_score_:.4f}  ({time.time()-t0:.1f}s)")

print("Tuning SVM (RBF)...")
svm_grid = GridSearchCV(
    SVC(kernel="rbf"),
    param_grid={"C": [0.1, 1, 10], "gamma": ["scale", 0.001, 0.01]},
    cv=cv, scoring="accuracy", n_jobs=-1
)
t0 = time.time()
svm_grid.fit(X_trainpool_scaled, y_trainpool)
print(f"  Best params: {svm_grid.best_params_}  CV acc: {svm_grid.best_score_:.4f}  ({time.time()-t0:.1f}s)")

print("Tuning Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={"n_estimators": [100, 200, 300], "max_depth": [None, 20, 40]},
    cv=cv, scoring="accuracy", n_jobs=-1
)
t0 = time.time()
rf_grid.fit(X_trainpool_scaled, y_trainpool)
print(f"  Best params: {rf_grid.best_params_}  CV acc: {rf_grid.best_score_:.4f}  ({time.time()-t0:.1f}s)")

# Save everything to JSON so it's never lost again
results = {
    "LogisticRegression": {
        "best_params": lr_grid.best_params_,
        "best_cv_accuracy": lr_grid.best_score_,
    },
    "SVM_RBF": {
        "best_params": svm_grid.best_params_,
        "best_cv_accuracy": svm_grid.best_score_,
    },
    "RandomForest": {
        "best_params": rf_grid.best_params_,
        "best_cv_accuracy": rf_grid.best_score_,
    },
}

with open(SAVE_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {SAVE_PATH}")
print(json.dumps(results, indent=2))
