import numpy as np

def softmax_function(X, W, b):
    dot_product = np.dot(X, W) + b  # Add the bias term
    stabilized_dot_product = dot_product - np.max(dot_product, axis=1, keepdims=True)
    exp_scores = np.exp(stabilized_dot_product)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probabilities

def softmax_loss(X, Y, W, b): 
    """
    Calculate the softmax loss for a given set of predictions, labels, and weights.

    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix where each row represents a single sample.
    Y : numpy.ndarray
        Label matrix where each row represents the one-hot encoded labels for a corresponding sample in X.
    W : numpy.ndarray
        Weight matrix corresponding to the features in X.
 
    Returns
    -------
    float
        The computed softmax loss averaged over all samples in X.

    Notes
    -----
    This function computes the negative log likelihood of the true labels, given the predictions made by softmax.
    """
    s_value = softmax_function(X, W, b) #get the softmax output (samples, classes)
    log_likelihood_times_Y = np.log(s_value)*Y
    loss =  -np.sum(log_likelihood_times_Y, axis=1, keepdims=True)  #return the softmax loss averaged over all samples in X ()
    loss = np.mean(loss)
    return loss

def softmax_loss_grad(X, Y, W, b):
    probabilities = softmax_function(X, W, b)
    grad_W = np.dot(X.T, probabilities - Y) / X.shape[0]
    grad_b = np.sum(probabilities - Y, axis=0) / X.shape[0]  # Sum across columns for biases
    return grad_W, grad_b

def softmax_cost_and_grad(X, Y, W, b):
    loss = softmax_loss(X, Y, W, b)
    grad_W, grad_b = softmax_loss_grad(X, Y, W, b)
    return loss, grad_W, grad_b