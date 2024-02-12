import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def tanh(Z):
    return np.tanh(Z)

def tanh_deriv(Z):
    return 1 - np.tanh(Z)**2

def softmax(Z):
    expZ = np.exp(Z - np.max(Z , axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

class NN:
    def __init__(self, input_size, output_size, hidden_size = 6, num_layers=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.A = None
        self.caches = None
        self.grads = None
        self.loss = 0
        self.parameters = {}
        self.initiate_parameters()

    def initiate_parameters(self):
        self.parameters['W1'] = np.random.randn(self.hidden_size, self.input_size) if self.num_layers > 1 else np.random.randn(self.output_size, self.input_size)
        self.parameters['b1'] = np.zeros((self.hidden_size, 1)) if self.num_layers > 1 else np.zeros((1, self.output_size))
        for i in range(2, self.num_layers):
            self.parameters['W' + str(i)] = np.random.randn(self.hidden_size, self.hidden_size)
            self.parameters['b' + str(i)] = np.zeros((self.hidden_size, 1))
        if self.num_layers > 1:
            self.parameters['W' + str(self.num_layers)] = np.random.randn(self.hidden_size, self.output_size)
        self.parameters['b' + str(self.num_layers)] = np.zeros((1, self.output_size))
        for key in self.parameters:
            if key[0] == 'W':
                self.parameters[key] = self.parameters[key] / np.linalg.norm(self.parameters[key])

    def forward_propagation(self, X):
        caches = {}
        caches['A0'] = X
        A = X
        for layer in range(1,self.num_layers):
            A, caches = self.hidden_layer_forward(A,layer,caches)

        A, caches = self.last_layer_forward(A, caches)

        self.A = A
        self.caches = caches

        return A

    def compute_loss(self, Y, A=None):
        if A is None:
            A = self.A
        log_likelihood = np.log(A) * Y
        loss = -np.sum(log_likelihood, axis=1, keepdims=True)
        loss = np.mean(loss)
        return loss # might need to average this loss
    
    def backward_propagation(self, Y):
        grads = {}

        dZ = self.A - Y
        A_prev = self.caches['A' + str(self.num_layers - 1)]

        samples = A_prev.shape[1]

        W_last = self.parameters['W' + str(self.num_layers)]
        dW_last = np.dot(A_prev, dZ)/samples
        db_last = np.sum(dZ, axis=0, keepdims=True)/samples

        grads['dW' + str(self.num_layers)] = dW_last
        grads['db' + str(self.num_layers)] = db_last

        dZ = np.dot(W_last, dZ.T)/samples

        for layer in reversed(range(1, self.num_layers)):
            A_prev = self.caches['A' + str(layer - 1)]
            Z = self.caches['Z' + str(layer)]

            dW = np.dot((tanh_deriv(Z) * dZ), A_prev.T) 
            db = np.sum(tanh_deriv(Z) * dZ, axis=1, keepdims=True)

            grads['dW' + str(layer)] = dW
            grads['db' + str(layer)] = db

            if layer > 1:
                W = self.parameters['W' + str(layer)]
                dZ = np.dot(W.T, (tanh_deriv(Z) * dZ))
    
        self.grads = grads
        return grads
    
    def hidden_layer_forward(self, A, layer,caches, perturb=None):
        W = self.parameters['W' + str(layer)]
        b = self.parameters['b' + str(layer)]
        Z = np.dot(W, A) + b
        A = tanh(Z)
        caches['Z' + str(layer)] = Z
        caches['A' + str(layer)] = A
        return A, caches
    
    def last_layer_forward(self, A, caches ,perturb=None, epsilon = 0.5):
        W = self.parameters['W' + str(self.num_layers)]
        b = self.parameters['b' + str(self.num_layers)]
        if perturb is not None:
            W = W + epsilon * perturb['W']
            b = b + epsilon * perturb['b']

        Z = np.dot(A.T, W) + b
        A = softmax(Z)
        caches['Z' + str(self.num_layers)] = Z
        caches['A' + str(self.num_layers)] = A
        return A, caches
    
    def last_layer_backward(self, Y, A, X, caches):
        grads = {}

        dZ = A - Y
        A_prev = X

        samples = A_prev.shape[1]

        dW_last = np.dot(A_prev, dZ)/samples
        db_last = np.sum(dZ, axis=0, keepdims=True)/samples

        grads['dW' + str(self.num_layers)] = dW_last
        grads['db' + str(self.num_layers)] = db_last
        return grads


def last_layer_gradient_check(epsilon = 0.5):
    
    epsilons = [epsilon**(i+1) for i in range(10)]
    X = np.array([[1, 2], [3, 4], [5, 6]]).T  # 3 samples, 2 features each
    Y = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot encoded labels for 3 samples, 2 classes
    input_size = 2
    hidden_size = 3
    output_size = 2
    num_layers = 1
    nn = NN(input_size, output_size, hidden_size, num_layers)

    A, caches = nn.last_layer_forward(X,{})
    original_loss = nn.compute_loss(Y, A)
    pertrubations = {}
    pertrubations['W'] = np.random.randn(*nn.parameters['W1'].shape)
    pertrubations['W'] /= np.linalg.norm(pertrubations['W'])
    pertrubations['b'] = np.random.randn(*nn.parameters['b1'].shape)
    pertrubations['b'] /= np.linalg.norm(pertrubations['b'])

    linear_error = []
    quadratic_error = []
    grads = nn.last_layer_backward(Y, A, X, caches)
    analytical_grads_w = np.dot(pertrubations['W'].flatten(), grads['dW1'].flatten())
    analytical_grads_b = np.dot(pertrubations['b'].flatten(), grads['db1'].flatten())
    total_analytical_grad = analytical_grads_w + analytical_grads_b

    for epsilon in epsilons:
        A, caches = nn.last_layer_forward(X, {}, perturb=pertrubations, epsilon=epsilon)
        loss = nn.compute_loss(Y, A)
        linear_error.append(np.abs(loss - original_loss))
        quadratic_error.append(np.abs(loss - original_loss - total_analytical_grad * epsilon))
        print(f"Epsilon: {epsilon}, Linear Error: {linear_error[-1]}, Quadratic Error: {quadratic_error[-1]}")
    
    plot_errors(epsilons, linear_error, quadratic_error, "last layer Gradient Test Errors (Including Bias)")

    return

def hidden_layer_jacobian_check(epsilon = 0.5):
    return

def full_gradient_check(epsilon = 0.5):
    epsilons = [epsilon**(i+1) for i in range(10)]
    X = np.array([[1, 2], [3, 4], [5, 6]]).T  # 3 samples, 2 features each
    Y = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot encoded labels for 3 samples, 2 classes
    input_size = 2
    hidden_size = 3
    output_size = 2
    num_layers = 3

    nn = NN(input_size, output_size, hidden_size, num_layers)
    A = nn.forward_propagation(X)
    original_loss = nn.compute_loss(Y)
    grads = nn.backward_propagation(Y)

    pertrubations = {}
    for layer in range(1, num_layers+1):
        pertrubations['W' + str(layer)] = np.random.randn(*nn.parameters['W' + str(layer)].shape)
        pertrubations['W' + str(layer)] /= np.linalg.norm(pertrubations['W' + str(layer)])
        pertrubations['b' + str(layer)] = np.random.randn(*nn.parameters['b' + str(layer)].shape)
        pertrubations['b' + str(layer)] /= np.linalg.norm(pertrubations['b' + str(layer)])
    
    linear_error = []
    quadratic_error = []
    total_analytical_grad = 0
    for layer in range(1, num_layers+1):
        analytical_grads_w = np.dot(pertrubations['W' + str(layer)].flatten(), grads['dW' + str(layer)].flatten())
        analytical_grads_b = np.dot(pertrubations['b' + str(layer)].flatten(), grads['db' + str(layer)].flatten())
        total_analytical_grad += analytical_grads_w + analytical_grads_b

    for epsilon in epsilons:
        for key in nn.parameters:
            nn.parameters[key] += epsilon * pertrubations[key]
        A = nn.forward_propagation(X)
        pertrubed_loss = nn.compute_loss(Y, A)
        linear_error.append(np.abs(pertrubed_loss - original_loss))
        quadratic_error.append(np.abs(pertrubed_loss - original_loss - total_analytical_grad * epsilon))
        print(f"Epsilon: {epsilon}, Linear Error: {linear_error[-1]}, Quadratic Error: {quadratic_error[-1]}")

        for key in nn.parameters:
            nn.parameters[key] -= epsilon * pertrubations[key]

    plot_errors(epsilons, linear_error, quadratic_error, "full network Gradient Test Errors")


def plot_errors(epsilons, linear_error, quadratic_error, title):
    iterations = [i for i in range(1, len(epsilons) + 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, linear_error, label='Linear Error')
    plt.plot(iterations, quadratic_error, label='Quadratic Error')
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def NNSGD(nn : NN, X, Y, learning_rate=0.1, num_iterations=100, mini_batch_size=30, RESULTS=True):
    losses = []
    for iter in range(num_iterations):
        indices = np.random.permutation(X.shape[1])
        shuffled_X = np.array(X[:, indices])
        shuffled_Y = np.array(Y[indices, :])
        current_loss = 0
        for i in range(0, X.shape[1], mini_batch_size):
            X_mini = shuffled_X[:, i:i+mini_batch_size]
            Y_mini = shuffled_Y[i:i+mini_batch_size, :]
           
            mini_batch_A = nn.forward_propagation(X_mini)
            current_loss += nn.compute_loss(Y_mini, mini_batch_A)
            grads = nn.backward_propagation(Y_mini)
        
            for layer in range(1, nn.num_layers+1):
                nn.parameters['W' + str(layer)] -= learning_rate * grads['dW' + str(layer)]
                nn.parameters['b' + str(layer)] -= learning_rate * grads['db' + str(layer)]
        loss = current_loss / (X.shape[1] / mini_batch_size)
        losses.append(loss)
        if RESULTS:
            print(f"Iteration: {iter}, Loss: {loss}")
    
    if RESULTS:
        plt.figure(figsize=(10, 6))
        plt.plot([i for i in range(num_iterations)], losses)
        plt.xlabel('iteration')
        plt.ylabel('Loss')
        plt.title('Loss over iterations')
        plt.grid(True)
        plt.show()
            
        

def main():
    choice = input("Enter 1 for NN test, 2 for last layer gradient check, 3 for full gradient check: ")
    if choice == "1" or choice == "3" or choice == "":   
        path = input("Enter the path to the data: ")
        if path == "":
            path = 'SwissRollData'
        mat = loadmat(path + '.mat')
        X = np.array(mat['Yt'])
        Y = np.array(mat['Ct']).T

        user_defined = input("Enter 1 to use user defined parameters, 0 to use default: ") == "1"

        

        input_size = X.shape[0]
        hidden_size = 6
        
        output_size = Y.shape[1]
        num_layers = 4

        if user_defined:
            hidden_layer_user_size = input("Enter the number of hidden neuron per layer (enter for default): ")
            layer_number_user_size = input("Enter the number of layers (enter for default): ")

            if hidden_layer_user_size != "":
                hidden_size = int(hidden_layer_user_size)
            if layer_number_user_size != "":
                num_layers = int(layer_number_user_size)

        #works great with 4 layers and 6 hidden units, 0.1 learning rate, 100 iterations
        nn = NN(input_size, output_size, hidden_size, num_layers)
        if choice == "1" or choice == "":
            whole_data = input("Enter 1 to train on the whole data, 2 to use 200 random data points: ")
            if whole_data == "2":
                indices = np.random.permutation(X.shape[1])
                X = np.array(X[:, indices[:200]])
                Y = np.array(Y[indices[:200], :])

            if user_defined:
                learning_rate = input("Enter the learning rate (enter for default): ")
                if learning_rate != "":
                    learning_rate = float(learning_rate)
                else:
                    learning_rate = 0.1

                num_iterations = input("Enter the number of iterations (enter for default): ")
                if num_iterations != "":
                    num_iterations = int(num_iterations)
                else:
                    num_iterations = 100
                
                mini_batch_size = input("Enter the mini batch size (enter for default): ")
                if mini_batch_size != "":
                    mini_batch_size = int(mini_batch_size)
                else:
                    mini_batch_size = 30
                
                NNSGD(nn, X, Y, learning_rate, num_iterations, mini_batch_size)
            else:
                NNSGD(nn, X, Y)
            
            #testing on the validation set
            X = np.array(mat['Yv'])
            Y = np.array(mat['Cv']).T
            A = nn.forward_propagation(X)
            loss = nn.compute_loss(Y)
            print(loss)

            # preds = np.argmax(A, axis=1)
            # true = np.argmax(Y, axis=1)
            # diff = np.abs(preds - true)
            # diff_indices = [i for i, x in enumerate(diff) if x == 1]
            # print(f"Number of misclassified samples: {len(diff_indices)}")
            # print("the indexes are: ", diff_indices)

            print("Predicted: ", np.argmax(A, axis=1))
            print("True: ", np.argmax(Y, axis=1))
            print(f"Accuracy: {np.mean(np.argmax(A, axis=1) == np.argmax(Y, axis=1))*100}%")
            return
        else:
            full_gradient_check()
            return
    
    elif choice == "2":

        last_layer_gradient_check()
        
        # X = np.array([[1, 2], [3, 4], [5, 6]]).T  # 3 samples, 2 features each
        # Y = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot encoded labels for 3 samples, 2 classes
        # input_size = 2
        # hidden_size = 6
        # output_size = 2
        # num_layers = 6
        # nn = NN(input_size, hidden_size, output_size, num_layers)
        # A = nn.forward_propagation(X)
        # loss = nn.compute_loss(Y)
        # print(loss)
        # grads = nn.backward_propagation(Y)
        # # print(grads)

if __name__ == "__main__":
    main()