import numpy as np
import matplotlib.pyplot as plt
from SGD import sgd


# The objective function and its gradient: Least Squares
def least_squares(X, y, w):
    predictions = X.dot(w)
    errors = predictions - y
    cost = (1/2) * np.mean(errors ** 2)
    gradient = X.T.dot(errors) / len(y)
    return cost, gradient


X = np.random.randn(100, 2)
w_true = np.array([1.5, -0.5])
y = X.dot(w_true) + np.random.randn(100)

# Parameters
learning_rate = 0.001
iterations = 10000
batch_size = 100  # Size of the minibatch

# Run SGD optimizer with minibatches
w_opt_minibatch, costs_minibatch = sgd(X, y, least_squares, learning_rate, iterations, batch_size)

# Plotting the optimization process with minibatches
plt.plot(costs_minibatch)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost reduction over iterations using SGD for Least Squares')
# plt.savefig('result_graphs/SGD_Least_Squares_Test.png')
plt.show()