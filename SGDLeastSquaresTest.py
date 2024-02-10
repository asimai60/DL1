import numpy as np
import matplotlib.pyplot as plt
from SGD import sgd
from least_squares import least_squares

#A small least squares example
#np.random.seed(0)



X = np.random.randn(100, 2)
w_true = np.array([1.5, -0.5])
y = X.dot(w_true) + np.random.randn(100) * 0.5 # Add some noise

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
plt.title('Cost reduction over iterations using SGD with minibatches')
# plt.savefig('SGDLeastSquaresTest.png')
plt.show()