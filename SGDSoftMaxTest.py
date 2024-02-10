import numpy as np
import matplotlib.pyplot as plt
from SGD import sgd
from SMaxReg import softmax_cost_and_grad
from scipy.io import loadmat

mat = loadmat('SwissRollData.mat')

X = np.array(mat['Yt']).T
y = np.array(mat['Ct']).T

# Parameters
learning_rate = 0.0001
iterations = 1000
batch_size = 100  # Size of the minibatch

# Run SGD optimizer with minibatches
w_opt_minibatch, biases, costs_minibatch = sgd(X, y, softmax_cost_and_grad, learning_rate, iterations, batch_size, bias = True)

# Plotting the optimization process with minibatches
# print(biases)
plt.plot(biases)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('bias adaption over iterations using SGD with minibatches')
# plt.savefig('SGDSoftMaxBiasesTest.png')
plt.show()

plt.plot(costs_minibatch)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost reduction over iterations using SGD with minibatches')
# plt.savefig('SGDSoftMaxTest.png')
plt.show()