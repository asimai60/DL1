import numpy as np
from SMaxReg import *
from matplotlib import pyplot as plt

def gradient_test(X, Y, W, b, epsilons):
    original_loss = softmax_loss(X, Y, W, b)

    # Generate a random perturbation vector for W and b
    dW = np.random.randn(*W.shape)
    dW /= np.linalg.norm(dW)  # Normalize dW to have a unit norm
    db = np.random.randn(*b.shape)
    db /= np.linalg.norm(db)  # Normalize db to have a unit norm

    linear_error = []
    quadratic_error = []
    for epsilon in epsilons:
        # Perturb W and b
        W_perturbed = W + epsilon * dW
        b_perturbed = b + epsilon * db

        # Calculate the new loss
        perturbed_loss = softmax_loss(X, Y, W_perturbed, b_perturbed)

        # Analytical gradient
        grad_W, grad_b = softmax_loss_grad(X, Y, W, b)
        analytical_grad_W = np.dot(dW.flatten(), grad_W.flatten())
        analytical_grad_b = np.dot(db.flatten(), grad_b.flatten())
        total_analytical_grad = analytical_grad_W + analytical_grad_b

        # Calculate differences
        linear_error.append(np.abs(perturbed_loss - original_loss))
        quadratic_error.append(np.abs(perturbed_loss - original_loss - epsilon * total_analytical_grad))

        print(f"Epsilon: {epsilon}, Linear Error: {linear_error[-1]}, Quadratic Error: {quadratic_error[-1]}")

    # Plotting
    iterations = [i for i in range(1, len(epsilons) + 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, linear_error, label='Linear Error')
    plt.plot(iterations, quadratic_error, label='Quadratic Error')
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('Error')
    plt.title('Gradient Test Errors (Including Bias)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example data (very simple and small for demonstration purposes)
X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features each
Y = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot encoded labels for 3 samples, 2 classes
W = np.random.randn(2, 2)  # Weight matrix initialized randomly (2 features, 2 classes)
b = np.random.randn(2)    # Bias vector initialized randomly (2 classes)

# Epsilon values for perturbation magnitude
epsilons = [0.5**(i+1) for i in range(10)]

# Perform the gradient test including bias
gradient_test(X, Y, W, b, epsilons)
