
## Linear Regression Categories,Encapsulated through categories (Class)

import numpy as np

class ManualLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iter = iterations
        self.theta = None
        self.cost_history = []

    def fit(self, X, y):

        #Train the model

        m = len(y)
        # Add intercept column (column of ones)
        X_final = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.zeros(X_final.shape[1])

        for _ in range(self.iter):
            predictions = np.dot(X_final, self.theta)
            errors = predictions - y
            gradient = (1 / m) * np.dot(X_final.T, errors)
            self.theta = self.theta - self.lr * gradient
            
            # Compute cost (Mean Squared Error)
            cost = (1 / (2 * m)) * np.sum((np.dot(X_final, self.theta) - y) ** 2)
            self.cost_history.append(cost)

    def predict(self, X):

        # Make predictions

        X_final = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X_final, self.theta)