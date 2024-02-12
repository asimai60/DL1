import numpy as np
from NN import *
from scipy.io import loadmat
import itertools
import csv

# Load the dataset
mat = loadmat('SwissRollData.mat')
X = np.array(mat['Yt'])
Y = np.array(mat['Ct']).T

input_size = X.shape[0]
output_size = Y.shape[1]
# Define the hyperparameters
hidden_layer_size = [np.arange(3,10,3)]
num_layers = [np.arange(3,6,1)]
learning_rate = [np.arange(0.01,1,0.1)]
num_iterations = [np.arange(100,500,100)]
mini_batch_size = [np.arange(10,200,20)]

# Create a dictionary to store the hyperparameters
hyperparameters = {'hidden_layer_size':hidden_layer_size, 'num_layers':num_layers, 'learning_rate':learning_rate, 'num_iterations':num_iterations, 'mini_batch_size':mini_batch_size}
all_possible_combinations = list(itertools.product(*[hidden_layer_size[0], num_layers[0], learning_rate[0], num_iterations[0], mini_batch_size[0]]))

reults = []
best_index = 0
best_yet = None
best_loss = np.inf
best_accuracy = 0

print('Total number of combinations:', )
for i in range(len(all_possible_combinations)):
    print('Training model with hyperparameters:', all_possible_combinations[i], f'({i+1}/{len(all_possible_combinations)})')
    nn = NN(input_size, output_size, all_possible_combinations[i][0], all_possible_combinations[i][1])
    NNSGD(nn, X, Y, all_possible_combinations[i][2], all_possible_combinations[i][3], all_possible_combinations[i][4], False)
    print('Training complete')

    X = np.array(mat['Yv'])
    Y = np.array(mat['Cv']).T
    A = nn.forward_propagation(X)
    loss = nn.compute_loss(Y)
    accuracy = np.mean(np.argmax(A, axis=1) == np.argmax(Y, axis=1))*100

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_loss = loss
        best_index = i
        best_yet = all_possible_combinations[i]

    reults.append((all_possible_combinations[i], loss, accuracy))

    print(f"Accuracy: {accuracy}%")
    print('Loss:', loss)
    print('-------------------------------------------------')
    print("best so far:", best_index,best_yet,"loss: ", best_loss,f"acc: {best_accuracy}%")
    print('-------------------------------------------------')

reults.sort(key=lambda x: x[2], reverse=True)
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Hyperparameters", "Loss", "Accuracy"])
    writer.writerows(reults)


    