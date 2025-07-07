import numpy as np

class MyLinearRegressionModel:
    def __init__(self, learning_rate=0.1, n_iterations=10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.beta = np.zeros(X_b.shape[1])
        for _ in range(self.n_iterations):
            y_pred = X_b.dot(self.beta)
            error = y - y_pred
            gradient = (2 / X_b.shape[0]) * X_b.T.dot(error)
            self.beta -= self.learning_rate * gradient

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.beta)

    def squared_error(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.mean((y - y_pred) ** 2) / np.var(y)
