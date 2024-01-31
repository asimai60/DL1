import numpy as np
import SGD

def objective_function(weights, x):
    return weights * x

weights = np.random.uniform(-5, 5)

data = np.random.uniform(-5, 5, 1000)
actual_data = objective_function(weights, data)

final_weights = SGD.sgd(weights, objective_function, data, epochs=50, Mb_amount=40, Lr=0.01)

print(f"Actual weights: {weights}" + f"\nFinal weights: {final_weights}")