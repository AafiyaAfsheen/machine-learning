import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay

digits = load_digits()
X = digits.data
y = digits.target
n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
y_score = svm_clf.predict_proba(X_test)

roc_auc = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
print(f"ROC-AUC (One-vs-Rest): {roc_auc:.4f}")

for i in range(3):
    RocCurveDisplay.from_predictions(y_test_bin[:, i], y_score[:, i], name=f"Class {i}")

plt.title("ROC Curves for Classes 0-2")
plt.show()
