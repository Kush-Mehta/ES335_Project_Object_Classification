import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from config import FEATURES_PATH, LABELS_PATH

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

# Held-out test set for later confidence-rejection analysis (20%)
X_trainpool, X_test, y_trainpool, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train pool: {X_trainpool.shape}, Test (held out): {X_test.shape}")

# Scale features (fit only on train pool, to avoid leaking test info)
scaler = StandardScaler()
X_trainpool_scaled = scaler.fit_transform(X_trainpool)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    scores = cross_val_score(model, X_trainpool_scaled, y_trainpool, cv=cv, scoring="accuracy")
    print(f"{name}: mean accuracy = {scores.mean():.4f}  (std = {scores.std():.4f})  folds = {scores}")
