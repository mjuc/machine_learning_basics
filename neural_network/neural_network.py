import numpy as np
import matplotlib.pyplot as plt

def sigmoid_in_point(x):
    return 1 / (1 + np.exp(-x))

def sigmoid(x):
    ret = np.empty(x.shape)
    iter = 0
    for i in x:
        ret[iter] = sigmoid_in_point(i)
        iter+=1
    return ret

def derivative_in_point(x):
    h = 0.001
    return (sigmoid_in_point(x+h)-sigmoid_in_point(x-h))/(2*h)

def sigmoid_derivative(x):
    ret = np.empty(x.shape)
    iter = 0
    for i in x:
        ret[iter]=derivative_in_point(i)
        iter+=1
    return ret

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
inputx = np.array([[0],[0],[1],[1]])
inputy = np.array([[0],[1],[1],[0]])


nw = NeuralNetwork(inputx,inputy)

nw.feedforward()
nw.backprop()

plt.plot(nw.output)

for x in range(0,1500):
    nw.feedforward()
    nw.backprop()

plt.plot(nw.output)