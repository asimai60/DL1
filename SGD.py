import numpy as np

# SGD
def sgd(X, y, objective_function : callable, learning_rate=0.01, iterations=100, minibatchsize=10, bias = False):
    w = np.random.rand(X.shape[1], y.shape[1]) if len(y.shape) > 1 else  np.random.rand(X.shape[1])
    w = w / np.linalg.norm(w)
    biases = []
    b = 0
    if bias:
        b = np.random.rand(y.shape[1])  # Initialize bias as zeros
        b = b / np.linalg.norm(b)
        biases = np.zeros((iterations, y.shape[1]))
    losses = []

    

    for epoch in range(iterations):
        

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        shuffled_data = X[indices]
        shuffled_labels = y[indices]
        loss = 0
        gradient = 0 # test
        for start_idx in range(0, X.shape[0], minibatchsize):
            end_idx = min(start_idx + minibatchsize, X.shape[0])
            X_batch = shuffled_data[start_idx:end_idx]
            y_batch = shuffled_labels[start_idx:end_idx]

            if bias:
                mini_batch_loss, grad_W, grad_b = objective_function(X_batch, y_batch, w, b)
            else:
                mini_batch_loss, grad_W = objective_function(X_batch, y_batch, w)
                grad_b = 0
            # print(grad_b)
            w -= learning_rate * grad_W  # Update weights
            b -= learning_rate * grad_b  # Update bias
            loss += mini_batch_loss
            
        if bias:
            biases[epoch] = b

        # Store the loss for this iteration after going through all the batches
        loss = loss / ((X.shape[0] / minibatchsize)+1)
        losses.append(loss)
        
    if bias:
        return w, biases, losses
    return w, losses

