import numpy as np

def softmax_function(X, W):
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
    The softmax function is applied to the dot product of X and W.
    """
    dot_product = np.dot(X,W) #(samples, classes)
    
    stabilized_dot_product = dot_product - (np.max(dot_product, axis=1, keepdims=True)) #stabilize the dot product to avoid overflow (samples, classes)
    exp_stabilized_dot_product = np.exp(stabilized_dot_product) #exponentiate the stabilized dot product (samples, classes)
    return exp_stabilized_dot_product / np.sum(exp_stabilized_dot_product, axis=1, keepdims=True) #return the softmax output (samples, classes)

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
    s_value = softmax_function(X, W) #get the softmax output (samples, classes)
    log_likelihood_times_Y = np.log(s_value)*Y
    return -np.sum(log_likelihood_times_Y) / X.shape[0] #return the softmax loss averaged over all samples in X ()

def softmax_loss_grad(X, Y, W):
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
    probabilities = softmax_function(X, W)
    return X.T.dot(probabilities - Y) / X.shape[0]

