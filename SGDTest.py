import numpy as np
import SGD

def objective_function(weights, x):
    return weights * x ** 2

weights = np.random.uniform(-1, 1)

data = np.random.uniform(-5, 5, 50)
actual_data = objective_function(weights, data)

final_weights = SGD.sgd(weights, objective_function, data, epochs=100, Mb_amount=8, Lr=0.1)

print(f"Actual weights: {weights}" + f"\nFinal weights: {final_weights}")