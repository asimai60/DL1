import numpy as np
import matplotlib.pyplot as plt
from SGD import sgd
from SMaxReg import softmax_cost_and_grad, softmax_function
from scipy.io import loadmat

mat = loadmat('SwissRollData.mat')

X = np.array(mat['Yt']).T
y = np.array(mat['Ct']).T

# Parameters
learning_rate = 0.001
iterations = 50
batch_size = 10  # Size of the minibatch

# Run SGD optimizer with minibatches
w_opt_minibatch, biases, costs_minibatch = sgd(X, y, softmax_cost_and_grad, learning_rate, iterations, batch_size, bias = True, plot_epoch_precents = True)

X_test = np.array(mat['Yv']).T
y_test = np.array(mat['Cv']).T
indices = np.arange(X_test.shape[0])
shuffled_data = X_test[indices]
shuffled_labels = y_test[indices]



plt.plot(costs_minibatch)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost reduction over iterations using SGD')
plt.savefig('result_graphs/SGD_Softmax_Test_cost_reduction.png')
plt.show()