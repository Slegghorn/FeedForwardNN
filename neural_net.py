import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1], [2]]).reshape(1,2)

def Relu(x):
    return x * (x > 0)
def dRelu(x):
    return 1 * (x > 0)

class NeuralNetwork():

    def __init__(self):
        self.w1 = np.random.normal(size = (2, 4))
        self.b1 = np.random.normal(size = (4))

        self.w2 = np.random.normal(size = (4))
        self.b2 = np.random.normal(size = (1))

    def feedforward(self, x):
        h = Relu(np.matmul(x, self.w1) + self.b1)
        return Relu(np.matmul(h, self.w2) + self.b2)

    def train(self, x_train, y_train):
        -1



nn = NeuralNetwork()
