import numpy as np
import matplotlib.pyplot as plt
from SMaxReg import softmax_function
from scipy.io import loadmat


# SGD
def sgd(X, y, objective_function : callable, learning_rate=0.01, iterations=100, minibatchsize=10, bias = False, plot_epoch_precents = False):


    w = np.random.rand(X.shape[1], y.shape[1]) if len(y.shape) > 1 else  np.random.rand(X.shape[1])
    w = w / np.linalg.norm(w)
    biases = []
    b = 0
    if bias:
        b = np.random.rand(y.shape[1])  # Initialize bias as zeros
        b = b / np.linalg.norm(b)
        biases = np.zeros((iterations, y.shape[1]))
    losses = []
    accuaracies = []
    test_accuracies = []

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
            
            w -= learning_rate * grad_W  # Update weights
            b -= learning_rate * grad_b  # Update bias
            loss += mini_batch_loss
            
        if bias:
            biases[epoch] = b

        # Store the loss for this iteration after going through all the batches
        loss = loss / ((X.shape[0] / minibatchsize)+1)
        losses.append(loss)

        if plot_epoch_precents:

            X_to_plot = shuffled_data[:100]
            Y_to_plot = shuffled_labels[:100]
            probabilities = softmax_function(X_to_plot, w, b)
            predicted_classes = np.argmax(probabilities, axis=1)
            true_classes = np.argmax(Y_to_plot, axis=1)
            accuracy = np.mean(predicted_classes == true_classes)
            accuaracies.append(accuracy)

            mat = loadmat('SwissRollData.mat')
            X_test = np.array(mat['Yv']).T
            y_test = np.array(mat['Cv']).T
            indices = np.arange(X_test.shape[0])
            np.random.shuffle(indices)
            shuffled_test_data = X_test[indices]
            shuffled_test_labels = y_test[indices]

            X_test_to_plot = shuffled_test_data[:100]
            Y_test_to_plot = shuffled_test_labels[:100]
            probabilities = softmax_function(X_test_to_plot, w, b)
            predicted_classes = np.argmax(probabilities, axis=1)
            true_classes = np.argmax(Y_test_to_plot, axis=1)
            test_accuracy = np.mean(predicted_classes == true_classes)
            test_accuracies.append(test_accuracy)


    if plot_epoch_precents:
        plt.plot(accuaracies)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'Acc of training data iters, Lr = {learning_rate}, mini-batch size = {minibatchsize}')
        plt.savefig('result_graphs/SGD_Softmax_trainingData_Test_accuracy.png')
        plt.show()

        plt.plot(test_accuracies)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'Acc of test data over iters, Lr = {learning_rate}, mini-batch size = {minibatchsize}')
        plt.savefig('result_graphs/SGD_Softmax_TestData_test_accuracy.png')
        plt.show()




        
    if bias:
        return w, biases, losses
    return w, losses

