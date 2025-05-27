import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = fetch_california_housing(as_frame=True)
df = data.frame

corr= df.corr(numeric_only=True)["MedHouseVal"].abs().sort_values(ascending=False)
feat= correlation[1:5].index.tolist()

X = df[feat]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Top 4 features:", feat)
print("MSE:", mse)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
