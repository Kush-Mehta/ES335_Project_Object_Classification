"""Per-class error analysis for two-stream late-fusion models."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from config import FEATURES_HOG_PATH, FEATURES_COLOR_PATH, LABELS_PATH, RESULTS_DIR

H = np.load(FEATURES_HOG_PATH)
C = np.load(FEATURES_COLOR_PATH)
y_raw = np.load(LABELS_PATH)
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = list(le.classes_)

H_train, H_test, C_train, C_test, y_train, y_test = train_test_split(
    H, C, y, test_size=0.2, stratify=y, random_state=42
)
scaler_h = StandardScaler().fit(H_train)
scaler_c = StandardScaler().fit(C_train)
H_tr = scaler_h.transform(H_train);  H_te = scaler_h.transform(H_test)
C_tr = scaler_c.transform(C_train);  C_te = scaler_c.transform(C_test)

with open(RESULTS_DIR / "best_params_twostream.json") as f:
    best = json.load(f)

def make_clf(name, params):
    if name == "LogisticRegression":
        return CalibratedClassifierCV(LogisticRegression(max_iter=1000, random_state=42, **params), cv=3)
    elif name == "SVM_RBF":
        return CalibratedClassifierCV(SVC(kernel="rbf", random_state=42, **params), cv=3)
    else:
        return RandomForestClassifier(random_state=42, **params)

MODEL_DISPLAY = {"LogisticRegression": "Logistic Regression", "SVM_RBF": "SVM RBF", "RandomForest": "Random Forest"}

out_lines = []

for name, info in best.items():
    alpha = info["alpha"]
    label = MODEL_DISPLAY[name]

    clf_h = make_clf(name, info["params"])
    clf_c = make_clf(name, info["params"])
    clf_h.fit(H_tr, y_train)
    clf_c.fit(C_tr, y_train)

    proba = alpha * clf_h.predict_proba(H_te) + (1 - alpha) * clf_c.predict_proba(C_te)
    preds = proba.argmax(axis=1)

    out_lines.append(f"\n{'='*60}")
    out_lines.append(f"{label}")
    out_lines.append(f"{'='*60}")
    out_lines.append(classification_report(y_test, preds, target_names=class_names))

    cm = confusion_matrix(y_test, preds)
    out_lines.append(f"{'Class':<12} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'FPR':>8} {'FNR':>8}")
    out_lines.append("-" * 58)
    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
        out_lines.append(f"{cls:<12} {TP:>6} {FP:>6} {FN:>6} {TN:>6} {FPR:>8.4f} {FNR:>8.4f}")

report = "\n".join(out_lines)
print(report)

out_path = RESULTS_DIR / "figures_v2" / "error_analysis_v2.txt"
with open(out_path, "w") as f:
    f.write(report)
print(f"\nSaved → {out_path}")