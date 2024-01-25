import numpy as np

def sgd(weights, gradients, Lr, data, epochs, Mb_amount):
    """
    parameters: 
        weights - the weight matrix
        gradient - vector of gradients of the loss function for every data point
        Lr - learning rate, hyperparameter
        data -
        epochs - number of iterations, hyperparameter
        Mb_amount - number of minibatches, hyperparameter


    output: improved weight matrix
    """
    SGD_weights = np.array()
    SGD_weights[0] = weights
    for k in range(epochs):
        indices = [0] + np.sort(np.random.choice(range(len(data), Mb_amount, False)))
        for j in range(len(Mb_amount)):
            current_grad = compute_minibatch_gradient(gradients,range(indices[j],indices[j+1]),SGD_weights[k])
            update_weights(current_grad, Lr, SGD_weights, k)
    return SGD_weights[epochs-1]


def compute_minibatch_gradient(gradients, minibatch_indices, current_weights):
    grad_vector = np.array()
    for i in minibatch_indices:
        for j in range(len(gradients[i])):
            grad_vector[j] += gradients[i](current_weights)/len(minibatch_indices)
    return grad_vector


def update_weights(gradient, Lr, weights, current_minibatch):
    weights[current_minibatch] = weights[current_minibatch] - Lr * gradient