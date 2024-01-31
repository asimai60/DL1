import numpy as np

def softmax(X, W):
    """
    Compute the softmax function for each row of the input x with the given weights W.

    Parameters
    ----------
    X : numpy.ndarray
        Input data matrix where each row represents a single sample.
    W : numpy.ndarray
        Weight matrix corresponding to the features in X.

    Returns
    -------
    numpy.ndarray
        The softmax output, where each row corresponds to the softmax calculation for a single sample in X.

    Notes
    -----
    The softmax function is applied to the dot product of the transpose of X and W.
    """
    dot_product = np.dot(X, W)
    stabilized_dot_product = dot_product - np.max(dot_product, axis=1, keepdims=True)
    exp_stabilized_dot_product = np.exp(stabilized_dot_product)
    return exp_stabilized_dot_product / np.sum(exp_stabilized_dot_product, axis=1, keepdims=True)

def softmax_loss(X, Y, W):
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
    return -np.sum(np.log(softmax(X, W))*Y) / X.shape[0]

def softmax_grad(X, Y, W):
    """
    Compute the gradient of the softmax loss function with respect to the weight matrix W.

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
    numpy.ndarray
        The gradient of the softmax loss with respect to W, averaged over all samples in X.

    Notes
    -----
    This function computes the gradient needed for updating the weight matrix W in gradient-based optimization algorithms.
    """
    probabilities = softmax(X, W)
    return X.dot(probabilities - Y) / X.shape[0]