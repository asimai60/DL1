import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt




def tanh(Z):
    return np.tanh(Z)

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e_Z / e_Z.sum(axis=0, keepdims=True)

def tanh_derivative(Z):
    return 1 - np.tanh(Z)**2



class NNinstance:
    def __init__(self, input_size, output_size, num_layers=3, hidden_layer_size=6):
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.parameters = self.initialize_parameters(input_size, hidden_layer_size, output_size, num_layers)
        self.A = None
        self.caches = None
        self.loss = None
        self.grads = None

    def initialize_parameters(self, input_size, hidden_layer_size=6, output_size=2, num_layers=3):
        
        np.random.seed(1)  # Ensure consistent initialization
        parameters = {}
        
        # Initialize weights and biases for all hidden layers
        for l in range(1, num_layers):
            parameters['W' + str(l)] = np.random.randn(hidden_layer_size, input_size if l == 1 else hidden_layer_size) * 0.01
            parameters['b' + str(l)] = np.zeros((hidden_layer_size, 1))
        
        # Initialize weights and biases for the output layer
        parameters['W' + str(num_layers)] = np.random.randn(output_size, hidden_layer_size) * 0.01
        parameters['b' + str(num_layers)] = np.zeros((output_size, 1))
        
        return parameters

    def forward_propagation(self, X):
        caches = {}
        caches['A0'] = X
        A = X
        for l in range(1, self.num_layers):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            
            Z = np.dot(W, A) + b
            A = tanh(Z)  # Use tanh for hidden layers
            
            caches['Z' + str(l)] = Z
            caches['A' + str(l)] = A
        
        # Output layer with softmax
        W = self.parameters['W' + str(self.num_layers)]
        b = self.parameters['b' + str(self.num_layers)]
        Z = np.dot(W, A) + b
        A = softmax(Z)  # Softmax for the output layer
        self.A = A
        
        caches['Z' + str(self.num_layers)] = Z
        caches['A' + str(self.num_layers)] = A
        self.caches = caches
    
        return self.A , caches
    
    def compute_nll_loss(self, Y):
        
        Y_encoded = np.eye(self.A.shape[0])[Y].T
        m = Y_encoded.shape[1]
        log_likelihood = -np.log(self.A[Y, range(m)])
        self.loss = np.sum(log_likelihood) / m

        return self.loss
    
    def backward_propagation(self, Y):
        grads = {}
        Y_encoded = np.eye(self.A.shape[0])[Y].T
        m = Y_encoded.shape[1]  # One-hot encode Y
        
        # Derivative of NLL Loss w.r.t. softmax inputs
        dZ = self.A - Y_encoded
        
        for l in reversed(range(1, self.num_layers + 1)):
            A_prev = self.caches['A' + str(l-1)] if l > 1 else self.caches['A0']
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                W = self.parameters['W' + str(l)]
                dZ = np.dot(W.T, dZ) * tanh_derivative(self.caches['Z' + str(l-1)])
            
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db
        
        self.grads = grads
        return self.grads
    
    
def NNsgd(NN : NNinstance ,X,Y, learning_rate=0.01, iterations=100, minibatchsize=10):
    losses = []
   
    for epoch in range(iterations):
        data = X.T
        labels = Y
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        shuffled_data = data[indices]
        shuffled_labels = labels[indices]
        
        loss = 0
        gradient = 0 # test
        for start_idx in range(0, X.shape[0], minibatchsize):
            mini_batch_loss = 0
            end_idx = min(start_idx + minibatchsize, X.shape[0])
            X_batch = shuffled_data[start_idx:end_idx].T
            y_batch = shuffled_labels[start_idx:end_idx]

            A, caches = NN.forward_propagation(X_batch)
            mini_batch_loss += NN.compute_nll_loss(y_batch)
            NN.backward_propagation(y_batch)

            for l in range(1, NN.num_layers):
                NN.parameters['W' + str(l)] -= learning_rate * NN.grads['dW' + str(l)]
                NN.parameters['b' + str(l)] -= learning_rate * NN.grads['db' + str(l)] 

            loss += mini_batch_loss
            

        # Store the loss for this iteration after going through all the batches
        loss = loss / ((X.shape[0] / minibatchsize)+1)
        losses.append(loss)
        
    return NN, losses



mat = loadmat('SwissRollData.mat')
X = np.array(mat['Yt'])
Y = np.array(mat['Ct']).T
Y = [int(i[1]) for i in Y]
Y = np.array(Y)
input_size = X.shape[0]  # Number of features
num_samples = X.shape[1]  # Number of samples

NN = NNinstance(input_size, output_size=2,num_layers=3)
NN, losses = NNsgd(NN, X, Y, learning_rate=0.01, iterations=100, minibatchsize=100)
# parameters = initialize_parameters(input_size, output_size=2,num_layers=10)  # Ensure output size matches number of classes
# A, caches = forward_propagation(X, parameters, num_layers=10)
# loss = compute_nll_loss(Y, A)
# grads = backward_propagation(Y, A, caches, parameters, num_layers=10)

print("NLL Loss:", losses)
# print("Gradients:")
# for key, value in NN.grads.items():
#     print(key,":\n", value)

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost reduction over iterations using SGD')
# plt.savefig('SGDSoftMaxTest.png')
plt.show()