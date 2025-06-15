from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=10000)
dt_clf = DecisionTreeClassifier(random_state=42)

log_reg.fit(X_train_scaled, y_train)
dt_clf.fit(X_train, y_train)

y_pred_logreg = log_reg.predict(X_test_scaled)
y_pred_tree = dt_clf.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-Score:", f1_score(y_true, y_pred))
    print()

evaluate_model("Logistic Regression", y_test, y_pred_logreg)
evaluate_model("Decision Tree Classifier", y_test, y_pred_tree)

print("Classification Report (Logistic Regression)")
print(classification_report(y_test, y_pred_logreg, target_names=data.target_names))

print("Classification Report (Decision Tree)")
print(classification_report(y_test, y_pred_tree, target_names=data.target_names))
