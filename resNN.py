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

class ResNN:
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
        for i in range(1, self.num_layers):
            self.parameters['W1_' + str(i)] = np.random.randn(self.input_size, self.input_size)
            self.parameters['W2_' + str(i)] = np.random.randn(self.input_size, self.input_size)
            self.parameters['b' + str(i)] = np.zeros((self.input_size, 1))
        if self.num_layers > 1:
            self.parameters['W' + str(self.num_layers)] = np.random.randn(self.input_size, self.output_size)
        self.parameters['b' + str(self.num_layers)] = np.zeros((1, self.output_size))
        for key in self.parameters:
            if key[0] == 'W':
                self.parameters[key] = self.parameters[key] / np.linalg.norm(self.parameters[key])

    def forward_propagation(self, X):
        caches = {}
        caches['A2_0'] = X
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
        A_prev = self.caches['A2_' + str(self.num_layers - 1)]

        samples = A_prev.shape[1]

        W_last = self.parameters['W' + str(self.num_layers)]
        dW_last = np.dot(A_prev, dZ)/samples
        db_last = np.sum(dZ, axis=0, keepdims=True)/samples

        grads['dW' + str(self.num_layers)] = dW_last
        grads['db' + str(self.num_layers)] = db_last

        dZ = np.dot(W_last, dZ.T)/samples
        for layer in reversed(range(1, self.num_layers)):
            A1_prev = self.caches['A2_' + str(layer - 1)]
            A2_prev = self.caches['A2_' + str(layer - 1)]
            Z_1 = self.caches['Z1_' + str(layer)]
            Z_2 = self.caches['Z1_' + str(layer)]
            W1 = self.parameters['W1_' + str(layer)]
            W2 = self.parameters['W2_' + str(layer)]

            dW1 = np.dot((tanh_deriv(Z_1) * np.dot(W2.T, dZ)),A2_prev.T)
            # print(dW1.shape)
            # print(dW1)
            dW2 = np.dot(dZ, tanh(Z_1).T)
            db = np.sum(tanh_deriv(Z_1) * np.dot(W2.T, dZ), axis=1, keepdims=True)

            grads['dW1_' + str(layer)] = dW1
            grads['dW2_' + str(layer)] = dW2
            grads['db' + str(layer)] = db

            if layer > 1:
                dZ = dZ + np.dot(W1.T, tanh_deriv(Z_1) * np.dot(W2.T, dZ))
    
        self.grads = grads
        return grads
    

    def hidden_layer_forward(self, A, layer,caches, perturb=None):
        W1 = self.parameters['W1_' + str(layer)]
        W2 = self.parameters['W2_' + str(layer)]
        b = self.parameters['b' + str(layer)]
        Z1 = np.dot(W1, A) + b
        A1 = tanh(Z1)
        Z2 = np.dot(W2, A1)
        A2 = A + Z2
        caches['Z1_' + str(layer)] = Z1
        caches['A1_' + str(layer)] = A1
        caches['Z2_' + str(layer)] = Z2
        caches['A2_' + str(layer)] = A2
        return A2, caches
    
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
    

def full_gradient_check(epsilon = 0.5):
    epsilons = [epsilon**(i+1) for i in range(10)]
    X = np.array([[1, 2], [3, 4], [5, 6]]).T  # 3 samples, 2 features each
    Y = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot encoded labels for 3 samples, 2 classes
    input_size = 2
    hidden_size = 3
    output_size = 2
    num_layers = 3

    nn = ResNN(input_size, output_size, hidden_size, num_layers)
    A = nn.forward_propagation(X)
    original_loss = nn.compute_loss(Y)
    grads = nn.backward_propagation(Y)

    pertrubations = {}
    for layer in range(1, num_layers):
        pertrubations['W1_' + str(layer)] = np.random.randn(*nn.parameters['W1_' + str(layer)].shape)
        pertrubations['W1_' + str(layer)] /= np.linalg.norm(pertrubations['W1_' + str(layer)])
        pertrubations['W2_' + str(layer)] = np.random.randn(*nn.parameters['W2_' + str(layer)].shape)
        pertrubations['W2_' + str(layer)] /= np.linalg.norm(pertrubations['W2_' + str(layer)])
        pertrubations['b' + str(layer)] = np.random.randn(*nn.parameters['b' + str(layer)].shape)
        pertrubations['b' + str(layer)] /= np.linalg.norm(pertrubations['b' + str(layer)])
    pertrubations['W' + str(num_layers)] = np.random.randn(*nn.parameters['W' + str(num_layers)].shape)
    pertrubations['W' + str(num_layers)] /= np.linalg.norm(pertrubations['W' + str(num_layers)])
    pertrubations['b' + str(num_layers)] = np.random.randn(*nn.parameters['b' + str(num_layers)].shape)
    pertrubations['b' + str(num_layers)] /= np.linalg.norm(pertrubations['b' + str(num_layers)])
    
    linear_error = []
    quadratic_error = []
    total_analytical_grad = 0
    for layer in range(1, num_layers):
        analytical_grads_w_1 = np.dot(pertrubations['W1_' + str(layer)].flatten(), grads['dW1_' + str(layer)].flatten())
        analytical_grads_w_2 = np.dot(pertrubations['W2_' + str(layer)].flatten(), grads['dW2_' + str(layer)].flatten())
        analytical_grads_b = np.dot(pertrubations['b' + str(layer)].flatten(), grads['db' + str(layer)].flatten())
        total_analytical_grad += analytical_grads_w_1 + analytical_grads_w_2 + analytical_grads_b
    analytical_grads_w = np.dot(pertrubations['W' + str(num_layers)].flatten(), grads['dW' + str(num_layers)].flatten())
    analytical_grads_b = np.dot(pertrubations['b' + str(num_layers)].flatten(), grads['db' + str(num_layers)].flatten())
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


def ResNNSGD(nn : ResNN, X, Y, learning_rate=0.1, num_iterations=100, mini_batch_size=30, RESULTS=True):
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
        
            for layer in range(1, nn.num_layers):
                nn.parameters['W1_' + str(layer)] -= learning_rate * grads['dW1_' + str(layer)]
                nn.parameters['W2_' + str(layer)] -= learning_rate * grads['dW2_' + str(layer)]
                nn.parameters['b' + str(layer)] -= learning_rate * grads['db' + str(layer)]
            nn.parameters['W' + str(nn.num_layers)] -= learning_rate * grads['dW' + str(nn.num_layers)]
            nn.parameters['b' + str(nn.num_layers)] -= learning_rate * grads['db' + str(nn.num_layers)]

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
        X = np.array([[1, 2], [3, 4], [5, 6]]).T  # 3 samples, 2 features each
        Y = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot encoded labels for 3 samples, 2 classes
        input_size = 2
        hidden_size = 6
        output_size = 2
        num_layers = 2
        nn = ResNN(input_size, output_size, hidden_size,  num_layers)
        A = nn.forward_propagation(X)
        # print(A)
        grads = nn.backward_propagation(Y)
        print(grads)

        full_gradient_check()

        mat = loadmat('SwissRollData.mat')
        X = np.array(mat['Yt'])
        Y = np.array(mat['Ct']).T

        input_size = X.shape[0]
        hidden_size = 6
        
        output_size = Y.shape[1]
        num_layers = 3

        nn = ResNN(input_size, output_size, hidden_size, num_layers)
        ResNNSGD(nn, X, Y, learning_rate=0.1, num_iterations= 100, mini_batch_size=30, RESULTS=True)
        X = np.array(mat['Yv'])
        Y = np.array(mat['Cv']).T
        A = nn.forward_propagation(X)
        loss = nn.compute_loss(Y)
        print(loss)


        print("Predicted: ", np.argmax(A, axis=1))
        print("True: ", np.argmax(Y, axis=1))
        print(f"Accuracy: {np.mean(np.argmax(A, axis=1) == np.argmax(Y, axis=1))*100}%")

if __name__ == "__main__":
    main()