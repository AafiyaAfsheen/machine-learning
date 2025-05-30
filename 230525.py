import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)

    def get_params(self):
        return {'intercept': self.weights[0], 'slope': self.weights[1:]}

if __name__ == "__main__":
    X = np.array([[1], [2], [4], [3], [5]])
    y = np.array([1, 3, 3, 2, 5])
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)

    print("Pred:", pred)
    print("Parameters:", model.get_params())
