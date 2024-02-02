import numpy as np

# The objective function and its gradient: Least Squares
def least_squares(X, y, w):
    predictions = X.dot(w)
    errors = predictions - y
    cost = (1/2) * np.mean(errors ** 2)
    gradient = X.T.dot(errors) / len(y)
    return cost, gradient