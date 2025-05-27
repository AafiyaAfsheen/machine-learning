import numpy as np
import pandas as pd

data = {
    'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Y': [3, 4, 2, 5, 6, 7, 8, 9, 10, 12]
}
df = pd.DataFrame(data)

X = df['X'].values
Y = df['Y'].values

n = len(X)
mean_x = np.mean(X)
mean_y = np.mean(Y)

numerator = sum((X - mean_x) * (Y - mean_y))
denominator = sum((X - mean_x) ** 2)
b1 = numerator / denominator
b0 = mean_y - b1 * mean_x

y_pred = b0 + b1 * X

mse = sum((Y - y_pred) ** 2) / n
rmse = np.sqrt(mse)
ss_total = sum((Y - mean_y) ** 2)
ss_residual = sum((Y - y_pred) ** 2)
r2_score = 1 - (ss_residual / ss_total)

print('MSE:', mse)
print('RMSE:', rmse)
print('R² Score:', r2_score)
