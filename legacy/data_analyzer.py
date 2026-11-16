import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FEATURES_PATH, LABELS_PATH
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

X_trainpool, X_test, y_trainpool, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_trainpool_scaled = scaler.fit_transform(X_trainpool)
X_test_scaled = scaler.transform(X_test)

# Final model: calibrated SVM so we get real probability outputs
base_svm = SVC(kernel="rbf")
model = CalibratedClassifierCV(base_svm, method="sigmoid", cv=5)
model.fit(X_trainpool_scaled, y_trainpool)

# --- Held-out test evaluation ---
y_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"Held-out test accuracy: {test_acc:.4f}")

classes = sorted(np.unique(y))
cm = confusion_matrix(y_test, y_pred, labels=classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(xticks_rotation=45, cmap="Blues")
plt.title("Confusion Matrix (SVM, held-out test set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# --- Confidence-threshold rejection analysis ---
probs = model.predict_proba(X_test_scaled)
confidences = np.max(probs, axis=1)
predicted_classes = model.classes_[np.argmax(probs, axis=1)]

thresholds = np.arange(0.0, 1.01, 0.05)
coverages = []
accuracies = []

for t in thresholds:
    accepted_mask = confidences >= t
    coverage = accepted_mask.mean()

    if accepted_mask.sum() == 0:
        acc = np.nan
    else:
        acc = accuracy_score(y_test[accepted_mask], predicted_classes[accepted_mask])

    coverages.append(coverage)
    accuracies.append(acc)

plt.figure(figsize=(7, 5))
plt.plot(thresholds, coverages, marker="o", label="Coverage")
plt.plot(thresholds, accuracies, marker="s", label="Accuracy on accepted items")
plt.xlabel("Confidence threshold")
plt.ylabel("Value")
plt.title("Accuracy vs. Coverage under Confidence-Threshold Rejection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rejection_curve.png", dpi=150)
plt.show()

for t, c, a in zip(thresholds, coverages, accuracies):
    print(f"threshold={t:.2f}  coverage={c:.3f}  accuracy={a if np.isnan(a) else round(a,3)}")
