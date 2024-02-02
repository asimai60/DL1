import numpy as np
from SMaxReg import *
from matplotlib import pyplot as plt

def gradient_test(X, Y, W, epsilons):
    # Calculate the original loss
    original_loss = softmax_loss(X, Y, W)

    
    
    # Generate a random perturbation vector for W
    d = np.random.randn(*W.shape)
    d /= np.linalg.norm(d) # Normalize d to have a unit norm
    linear_error = []
    quadratic_error = []
    for i, epsilon in enumerate(epsilons):
        # Perturb W
        W_perturbed = W + epsilon * d
        
        # Calculate the new loss
        perturbed_loss = softmax_loss(X, Y, W_perturbed)
        
        
        # Analytical gradient
        grad_W = softmax_loss_grad(X, Y, W)
        analytical_grad = np.dot(d.flatten(), grad_W.flatten())
        
        # Calculate differences

        linear_error.append(np.abs(perturbed_loss - original_loss))
        quadratic_error.append(np.abs(perturbed_loss - original_loss - epsilon * analytical_grad))
        
        print(f"Grad: {epsilon*analytical_grad}, Epsilon: {epsilon}, linear error: {linear_error[i]}, quadratic error: {quadratic_error[i]}")

    iterations = [i for i in range(1, len(epsilons) + 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, linear_error, label='Linear Error')
    plt.plot(iterations, quadratic_error, label='Quadratic Error')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Gradient Test Errors')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example data (very simple and small for demonstration purposes)
X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features each
Y = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot encoded labels for 3 samples, 2 classes
W = np.random.randn(2, 2)  # Weight matrix initialized randomly (2 features, 2 classes)

# Epsilon values
epsilons = [0.5**(i+1) for i in range(10)]

# Perform the gradient test
gradient_test(X, Y, W, epsilons)