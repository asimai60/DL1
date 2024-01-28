import numpy as np

def sgd(weights, objective_function, data, epochs=10, Mb_amount=10, Lr=0.1):
    """
    parameters: 
        weights - the weight matrix
        objective_function - the function that we are trying to approximate
        data - the observations that we are using to train the model 
        epochs - number of iterations, hyperparameter
        Mb_amount - number of minibatches, hyperparameter
        Lr - learning rate, hyperparameter

    output: improved weight matrix
    """
    SGD_weights_shape = epochs, weights.shape if type(weights) == np.ndarray else 1
    SGD_weights = np.zeros(SGD_weights_shape)
    SGD_weights[0] = weights
    for k in range(epochs):
        rng = np.random.default_rng(seed=None)

        shuffled_data = rng.permutation(data)

        indices = np.concatenate((np.array([0]), np.sort(np.random.choice(range(len(shuffled_data)), Mb_amount, False))))
        for j in range(Mb_amount):
            current_grad = compute_minibatch_gradient(objective_function, data, range(indices[j],indices[j+1]),SGD_weights[k])
            update_weights(current_grad, Lr, SGD_weights, k)
    return SGD_weights[epochs-1]


def compute_minibatch_gradient(objective_function, data, minibatch_indices, current_weights):
    """
    parameters:
        objective_function - the function that we are trying to approximate
        minibatch_indices - indices of the current minibatch
        current_weights - current weight matrix

    output: gradient vector of the loss function for the current minibatch
    """
    grad_vector_shape = len(current_weights) if type(current_weights) == np.ndarray else 1
    grad_vector = np.zeros(grad_vector_shape)
    for i in minibatch_indices:
        for j in range(len(current_weights)):
            grad_vector[j] += calculate_gradient_at_point(objective_function, data[i], current_weights)/len(minibatch_indices)
    return grad_vector


def update_weights(gradient, Lr, SGD_weights, current_iteration):
    """
    parameters:
        gradient - gradient vector of the loss function for the current minibatch
        Lr - learning rate, hyperparameter
        weights - current weight matrix
        current_minibatch - current minibatch indices

    output: updated weight matrix
    """
    SGD_weights[current_iteration] = SGD_weights[current_iteration] - Lr * gradient


def calculate_gradient_at_point(objective_function, point, current_weights):
    """
    parameters:
        objective_function - the function that we are trying to approximate
        point - current data point
        current_weights - current weight matrix

    output: gradient vector of the loss function for the current data point and weights
    """
    objective_gradient = np.zeros(len(current_weights) if type(current_weights) == np.ndarray else 1)
    epsilon = 0.0001
    for i in range(len(current_weights) if type(current_weights) == np.ndarray else 1):
        ith_coordinate = np.zeros(len(current_weights) if type(current_weights) == np.ndarray else 1)
        ith_coordinate[i] = 1
        objective_gradient[i] = objective_function(current_weights + ith_coordinate*epsilon, point) - objective_function(current_weights, point)
    return objective_gradient/epsilon