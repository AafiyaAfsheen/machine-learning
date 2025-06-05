import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data
y = data.target

np.random.seed(42)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

K = 5
fold_size = len(X) // K
folds_X = [X[i*fold_size:(i+1)*fold_size] for i in range(K)]
folds_y = [y[i*fold_size:(i+1)*fold_size] for i in range(K)]

accuracies = []

for i in range(K):
    X_test = folds_X[i]
    y_test = folds_y[i]

    X_train = np.vstack([folds_X[j] for j in range(K) if j != i])
    y_train = np.hstack([folds_y[j] for j in range(K) if j != i])

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Fold {i+1} Accuracy: {acc:.4f}")

average_accuracy = np.mean(accuracies)
print(f"\nAverage Accuracy across all 5 folds: {average_accuracy:.4f}")
