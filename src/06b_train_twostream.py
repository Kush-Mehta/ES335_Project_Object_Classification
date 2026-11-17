"""Two-stream late-fusion baseline: separate HOG and color classifiers, blended by alpha."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from config import FEATURES_HOG_PATH, FEATURES_COLOR_PATH, LABELS_PATH, RESULTS_DIR

# ── Load ──────────────────────────────────────────────────────────────────────
H = np.load(FEATURES_HOG_PATH)
C = np.load(FEATURES_COLOR_PATH)
y_raw = np.load(LABELS_PATH)

le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Classes:", list(le.classes_))
print(f"HOG: {H.shape}  Color: {C.shape}  Labels: {y.shape}\n")

# ── Train/test split (same seed as old pipeline) ──────────────────────────────
H_train, H_test, C_train, C_test, y_train, y_test = train_test_split(
    H, C, y, test_size=0.2, stratify=y, random_state=42
)

# ── Scale each stream independently (fit on train only) ──────────────────────
scaler_h = StandardScaler().fit(H_train)
scaler_c = StandardScaler().fit(C_train)

H_train_s = scaler_h.transform(H_train)
C_train_s = scaler_c.transform(C_train)
H_test_s  = scaler_h.transform(H_test)
C_test_s  = scaler_c.transform(C_test)

# ── Helper: late-fusion accuracy over CV folds ───────────────────────────────
def fusion_cv_accuracy(clf_h, clf_c, alpha, X_h, X_c, y, cv=5):
    """
    alpha=1.0 → HOG only, alpha=0.0 → color only, 0<alpha<1 → blend.
    Both classifiers are fit inside each fold independently.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    accs = []
    for tr, va in skf.split(X_h, y):
        sh = StandardScaler().fit(X_h[tr])
        sc = StandardScaler().fit(X_c[tr])
        clf_h.fit(sh.transform(X_h[tr]), y[tr])
        clf_c.fit(sc.transform(X_c[tr]), y[tr])
        ph = clf_h.predict_proba(sh.transform(X_h[va]))
        pc = clf_c.predict_proba(sc.transform(X_c[va]))
        preds = np.argmax(alpha * ph + (1 - alpha) * pc, axis=1)
        accs.append(accuracy_score(y[va], preds))
    return np.mean(accs), np.std(accs)

# ── Baseline CV with default hyperparams, alpha=0.5 ─────────────────────────
print("=" * 60)
print("Baseline 5-fold CV (default hyperparams, alpha=0.5)")
print("=" * 60)

models = {
    "LogisticRegression": (
        CalibratedClassifierCV(LogisticRegression(max_iter=1000, random_state=42), cv=3),
        CalibratedClassifierCV(LogisticRegression(max_iter=1000, random_state=42), cv=3),
    ),
    "SVM_RBF": (
        CalibratedClassifierCV(SVC(kernel="rbf", random_state=42), cv=3),
        CalibratedClassifierCV(SVC(kernel="rbf", random_state=42), cv=3),
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=100, random_state=42),
        RandomForestClassifier(n_estimators=100, random_state=42),
    ),
}

for name, (clf_h, clf_c) in models.items():
    mu, sd = fusion_cv_accuracy(clf_h, clf_c, alpha=0.5,
                                 X_h=H_train, X_c=C_train, y=y_train)
    print(f"  {name:20s}  CV acc = {mu:.4f} ± {sd:.4f}")

print()