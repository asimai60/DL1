import numpy as np
import SMaxReg as f

def gradient_test(x : np.ndarray):
    np.random.seed(1)
    d = np.random.randint(low=0,high=2, size = (x.shape[0]),)
    epsilon = 0.5
    
    class_amount = 10
    W = np.random.rand(class_amount,x.shape[0])
    #Y = np.eye(class_amount)[np.random.choice(class_amount, x.shape[0])]
    Y = np.zeros(class_amount)
    Y[0] = 1
    for iter in range(5):
        softmax_pertrubed_value = f.softmax_loss(x + epsilon*d,Y,W)
        softmax_value = f.softmax_loss(x,Y,W)
        grad_object = (epsilon*d).dot(f.softmax_loss_grad(x,Y,W))

        linear_error = np.abs(softmax_pertrubed_value - softmax_value)
        quadratic_error = np.abs(linear_error - grad_object)
        epsilon *= 0.5
        print(f"linear error: {linear_error}, quadratic error: {quadratic_error}")

x = np.array([2,4,3,7,5])
gradient_test(x)