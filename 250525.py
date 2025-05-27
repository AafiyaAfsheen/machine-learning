import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("C:\Users\HUAWEI\machine-learning\auto-mpg.csv")
df = df.replace("?", np.nan)
df["horsepower"] = pd.to_numeric(df["horsepower"])
df.dropna(subset=["horsepower", "mpg"], inplace=True)

X = df[["horsepower"]].values
y = df["mpg"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

degrees = [1, 2, 3]
models = {}
mse_scores = {}
x_range = np.linspace(X_scaled.min(), X_scaled.max(), 500).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled, y, color='gray', alpha=0.4, label="Data")

for d in degrees:
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_scores[d] = mean_squared_error(y_test, y_pred)
    y_curve = model.predict(x_range)
    plt.plot(x_range, y_curve, label=f"Degree {d} (MSE={mse_scores[d]:.2f})")

plt.title("Polynomial Regression: MPG vs Horsepower")
plt.xlabel("Normalized Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.show()
