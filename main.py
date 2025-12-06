import numpy as np
from tensors import Tensor

x = np.array([1,2,3,4])
y_training = np.array([3,6,9,12])

# Random number
w = Tensor(np.random.randn())


# Finds the error/loss
def mse(y_training, y_pred): 
    error = y_pred - y_training
    squared = error ** 2
    avg = np.mean(squared)
    return avg

# Finds gradient. Which direction to move to find value
def grad(y_training, y_pred, x):
    error = (y_pred - y_training) * x
    avg = np.mean(2 * error)
    return avg

# Learning rate
lr = 0.1

for i in range(50):
    y_pred = w * x
    gradient = grad(y_training, y_pred, x) # Find gradient
    w -= lr * gradient # Update parameter
    loss = mse(y_training, y_pred) # Finds loss; how fare 
    print(f"Step {i}: w={w:.3f}, loss={loss:.3f}")