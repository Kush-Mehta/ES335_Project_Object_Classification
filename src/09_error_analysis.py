import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

from config import FEATURES_PATH, LABELS_PATH, PARAMS_PATH, FIGURES_DIR
OUT_PATH = FIGURES_DIR / "error_analysis.txt"

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

X_trainpool, X_test, y_trainpool, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_trainpool_scaled = scaler.fit_transform(X_trainpool)
X_test_scaled      = scaler.transform(X_test)

with open(PARAMS_PATH) as f:
    params = json.load(f)

classes = sorted(np.unique(y))

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, **params["LogisticRegression"]["best_params"]
    ),
    "SVM RBF": CalibratedClassifierCV(
        SVC(kernel="rbf", **params["SVM_RBF"]["best_params"]),
        method="sigmoid", cv=5
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42, **params["RandomForest"]["best_params"]
    ),
}

report_lines = []

for model_name, model in models.items():
    print(f"Fitting {model_name}...")
    model.fit(X_trainpool_scaled, y_trainpool)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    header = f"\n{'='*60}\n{model_name}  |  Test Acc: {acc:.4f}  |  Macro F1: {macro_f1:.4f}\n{'='*60}"
    print(header)
    report_lines.append(header)

    # Full precision/recall/F1 per class
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)
    print(report)
    report_lines.append(report)

    # Per-class TP / FP / FN / TN breakdown
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    detail_header = f"{'Class':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'FPR':>8} {'FNR':>8}"
    print(detail_header)
    report_lines.append(detail_header)

    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP   # predicted as this class but wrong
        FN = cm[i, :].sum() - TP   # actually this class but missed
        TN = cm.sum() - TP - FP - FN

        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # false positive rate
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # false negative rate (miss rate)

        line = f"{cls:<15} {TP:>6} {FP:>6} {FN:>6} {TN:>6} {FPR:>8.4f} {FNR:>8.4f}"
        print(line)
        report_lines.append(line)

# Save full report
with open(OUT_PATH, "w") as f:
    f.write("\n".join(report_lines))
print(f"\nFull error analysis saved to {OUT_PATH}")
