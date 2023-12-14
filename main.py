import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles 

#--------------------------------------------------------------------------------------------------------------------------------------------------------

# Neural layers class

class layer():
    def __init__(self, n_con, neurals, act_fun):
        self.w = np.random.rand(n_con, neurals) * 2 - 1
        self.b = np.random.rand(1, neurals) * 2 - 1
        self.neurals = neurals
        self.act_fun = act_fun

#--------------------------------------------------------------------------------------------------------------------------------------------------------

# Functions

def create_network_neural(cant):

    neural_struc = []

    for i in range(cant):
        print()
        if i == 0:
            neural_struc.append(layer(p, 2, sig_act))
        if i == 1:
            neural_struc.append(layer(neural_struc[i-1].neurals, 3, sig_act))
        if i == 2:
            neural_struc.append(layer(neural_struc[i-1].neurals, 2, sig_act))
        if i == 3:
            neural_struc.append(layer(neural_struc[i-1].neurals, 1, sig_act))            

    return neural_struc

def train(neural_net, x, y, loss_f, lr = 0.1):
    # Forward pass
    output = [(None, x)]

    for i in range(len(neural_net)):
        z = output[-1][1] @ neural_net[i].w + neural_net[i].b
        a = neural_net[i].act_fun[0](z)
        output.append((z, a))

    # Backward
    deltas = []

    for i in reversed(range(0, len(neural_net))):
        z = output[i+1][0]
        a = output[i+1][1]

        if i == len(neural_net) - 1:
            deltas.insert(0, loss_f[1](a, y) * neural_net[i].act_fun[1](a)) 
        else:
            deltas.insert(0, deltas[0] @ _w.T * neural_net[i].act_fun[1](a)) 
        _w = neural_net[i].w

    # Gradient descent
    neural_net[i].b = neural_net[i].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
    neural_net[i].w = neural_net[i].w - output[i][1].T @ deltas[0] * lr

#--------------------------------------------------------------------------------------------------------------------------------------------------------

# Initialization of variables

np.random.seed(42)
n = 500  # Number of data points
p = 2      # Number of inputs for the first layer
x, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
y = y[:, np.newaxis]

# Test set percentage
test_percentage = 0.2

# Number of data points for the test set
num_test = int(n * test_percentage)

# Indices for the test set
test_indices = np.random.choice(n, num_test, replace=False)

# Indices for the training set
train_indices = np.setdiff1d(np.arange(n), test_indices)

# Test sets
x_test, y_test = x[test_indices], y[test_indices]

# Training sets
x_train, y_train = x[train_indices], y[train_indices]

#--------------------------------------------------------------------------------------------------------------------------------------------------------

# Activation functions

sig_act = (lambda x: 1 / (1 + np.e**(-x)),
            lambda x:  (np.e**x)/(1+np.e**x)**2)

relu_act = (lambda x: np.maximum(0, x),
            lambda x: np.where(x > 0, 1, 0))

#--------------------------------------------------------------------------------------------------------------------------------------------------------

# Error functions

cuadra = (lambda yp, yr: np.mean((yp - yr)**2),
            lambda yp, yr: yp - yr)

#--------------------------------------------------------------------------------------------------------------------------------------------------------

# Training the neural network

errors_train = []

neural_struc = create_network_neural(4)

for l in range(1000):
    train(neural_struc, x_train, y_train, cuadra)
    if l % 100 == 0:
        predictions_train = neural_struc[-1].act_fun[0](x_train @ neural_struc[0].w + neural_struc[0].b)
        error_train = cuadra[0](predictions_train, y_train)
        errors_train.append(error_train)
#--------------------------------------------------------------------------------------------------------------------------------------------------------

# Evaluation on the test set

predictions_test = neural_struc[-1].act_fun[0](x_test @ neural_struc[0].w + neural_struc[0].b)
error_test = cuadra[0](predictions_test, y_test)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First plot: Scatter plot of the data
ax1.scatter(x[y[:, 0] == 0, 0], x[y[:, 0] == 0, 1], c="orange", label="Class 0")
ax1.scatter(x[y[:, 0] == 1, 0], x[y[:, 0] == 1, 1], c="blue", label="Class 1")
ax1.set_title('Scatter Plot of the Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()
ax1.axis("equal")

# Second plot: Training error plot
ax2.plot(errors_train, label='Training Error')
ax2.axhline(y=error_test, color='r', linestyle='--', label='Test Error')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Error')
ax2.legend()
ax2.set_title('Training and Test Error Over Iterations')

# Show both plots
plt.tight_layout()
plt.show()






