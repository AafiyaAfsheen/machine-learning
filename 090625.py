import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr = LogisticRegression(max_iter=10000, solver='liblinear')
dt = DecisionTreeClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

def rank_models(models, X_test, y_test, metric):
    metric_func = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'roc_auc': roc_auc_score
    }

    if metric not in metric_func:
        raise ValueError(f"Invalid metric. Choose from: {list(metric_func.keys())}")

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)

        if metric == 'roc_auc':
            if hasattr(model, "predict_proba"):
                score = metric_func[metric](y_test, model.predict_proba(X_test)[:, 1])
            else:
                score = np.nan  
        else:
            score = metric_func[metric](y_test, y_pred)

        results.append({'Model': name, metric: score})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=metric, ascending=False).reset_index(drop=True)
    return results_df

models = {
    'Logistic Regression': lr,
    'Decision Tree': dt
}

print("Ranking based on Accuracy:\n", rank_models(models, X_test, y_test, 'accuracy'))
print("\nRanking based on F1 Score:\n", rank_models(models, X_test, y_test, 'f1'))
print("\nRanking based on ROC AUC:\n", rank_models(models, X_test, y_test, 'roc_auc'))
