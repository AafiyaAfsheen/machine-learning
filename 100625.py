import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": {
        "model": LogisticRegression(),
        "params": {"C": [0.01, 0.1, 1, 10]}
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {"max_depth": [3, 5, 10, None]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [50, 100], "max_depth": [5, 10]}
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(),
        "params": {"learning_rate": [0.01, 0.1], "n_estimators": [100, 150]}
    },
    "Support Vector Machine": {
        "model": SVC(probability=True),
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7]}
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "params": {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1]}
    }
}

results = []

for name, config in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", config["model"])
    ])
    
    clf = GridSearchCV(pipe, {"clf__" + k: v for k, v in config["params"].items()}, 
                       cv=5, scoring='f1', n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Best Params": clf.best_params_,
        "F1 Score": f1,
        "AUC-ROC": auc
    })

leaderboard = pd.DataFrame(results).sort_values(by=["F1 Score", "AUC-ROC"], ascending=False).reset_index(drop=True)
print("\nFinal Model Leaderboard:")
print(leaderboard)
