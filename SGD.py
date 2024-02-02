import numpy as np

def sgd(weights, objective_function, data, epochs=10, Mb_amount=10, Lr=0.1):
    """
    Perform Stochastic Gradient Descent (SGD) optimization.

    Parameters:
    - weights (numpy.ndarray): Initial weights for the optimization.
    - objective_function (callable): The objective function to be minimized.
    - data (numpy.ndarray): Data used for the optimization process.
    - epochs (int, optional): Number of complete passes through the dataset. Default is 10.
    - Mb_amount (int, optional): Number of minibatches for SGD. Default is 10.
    - Lr (float, optional): Learning rate for the optimization. Default is 0.1.

    Returns:
    - numpy.ndarray: The optimized weights after 'epochs' iterations.
    
    Notes:
    - The function updates weights based on the gradient computed for each minibatch.
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
        current_loss = compute_loss(objective_function, data, SGD_weights[k])
    return SGD_weights[epochs-1]


def compute_minibatch_gradient(objective_function, data, minibatch_indices, current_weights):
    """
    Compute the gradient vector for a minibatch.

    Parameters:
    - objective_function (callable): The objective function whose gradient is to be computed.
    - data (numpy.ndarray): The dataset used in the optimization process.
    - minibatch_indices (list or numpy.ndarray): Indices of the data points in the current minibatch.
    - current_weights (numpy.ndarray): Current weights of the model.

    Returns:
    - numpy.ndarray: The gradient vector for the specified minibatch.
    """

    grad_vector_shape = len(current_weights) if type(current_weights) == np.ndarray else 1
    grad_vector = np.zeros(grad_vector_shape)
    for i in minibatch_indices:
        for j in range(len(current_weights)):
            grad_vector[j] += calculate_gradient_at_point(objective_function, data[i], current_weights)/len(minibatch_indices)
    return grad_vector


def update_weights(gradient, Lr, SGD_weights, current_iteration):
    """
    Update the weights for the current iteration in SGD.

    Parameters:
    - gradient (numpy.ndarray): Gradient vector used for updating the weights.
    - Lr (float): Learning rate.
    - SGD_weights (numpy.ndarray): Array storing the weights at each iteration.
    - current_iteration (int): The current iteration number in the optimization process.

    Effects:
    - Modifies the SGD_weights array in-place to store the updated weights for the current iteration.
    - Prints the current weights for monitoring purposes.
    """
     
    SGD_weights[current_iteration] = SGD_weights[current_iteration] - Lr * gradient
    print(f"current ({current_iteration}) weights: " + str(SGD_weights[current_iteration]))


def calculate_gradient_at_point(objective_function, point, current_weights):
    """
    Calculate the gradient of the objective function at a given point.

    Parameters:
    - objective_function (callable): The objective function for which the gradient is calculated.
    - point (numpy.ndarray): The point at which the gradient is calculated.
    - current_weights (numpy.ndarray): Current weights used in the calculation of the gradient.

    Returns:
    - numpy.ndarray: The gradient of the objective function at the specified point.

    Notes:
    - The function uses a finite difference method to approximate the gradient.
    - Prints the gradient at the specified point for monitoring purposes.
    """

    objective_gradient = np.zeros(len(current_weights) if type(current_weights) == np.ndarray else 1)
    epsilon = 0.000001
    for i in range(len(current_weights) if type(current_weights) == np.ndarray else 1):
        ith_coordinate = np.zeros(len(current_weights) if type(current_weights) == np.ndarray else 1)
        ith_coordinate[i] = 1
        objective_gradient[i] = objective_function(current_weights + ith_coordinate*epsilon, point) - objective_function(current_weights, point)
    print(f"gradient at point {point}: " + str(objective_gradient/epsilon))
    return objective_gradient/epsilon

def compute_loss(objective_function, data, current_weights):
    """
    Compute the loss function for the current weights.

    Parameters:
    - objective_function (callable): The objective function for which the loss is calculated.
    - data (numpy.ndarray): The dataset used in the optimization process.
    - current_weights (numpy.ndarray): Current weights used in the calculation of the loss.

    Returns:
    - float: The loss function for the current weights.
    """

    loss = 0
    for i in range(len(data)):
        loss += objective_function(current_weights, data[i])
    return loss/len(data)