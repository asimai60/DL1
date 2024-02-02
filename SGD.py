import numpy as np
from least_squares import least_squares

# SGD
def sgd(X, y, objective_function : callable, learning_rate=0.01, iterations=100, minibatchsize=10):
    w = np.zeros(X.shape[1])
    costs = []

    for epoch in range(iterations):
        

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        shuffled_data = X[indices]
        shuffled_labels = y[indices]
        
        
        for start_idx in range(0, X.shape[0], minibatchsize):
            end_idx = min(start_idx + minibatchsize, X.shape[0])
            X_batch = shuffled_data[start_idx:end_idx]
            y_batch = shuffled_labels[start_idx:end_idx]

            # Compute cost and gradient for the batch
            cost, gradient = objective_function(X_batch, y_batch, w)
            w -= learning_rate * gradient  # Update weights

        # Store the cost for this iteration after going through all the batches
        costs.append(cost)

    return w, costs

