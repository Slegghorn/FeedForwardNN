import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 0]]).reshape(1,2)
y = np.array([[1]]).reshape(1, 1)

lr = 0.03
epochs = 1000


def Relu(x):
    return x * (x > 0)
def dRelu(x):
    return 1 * (x > 0)

def sig(x):
    return 1 / (1 + np.exp(-x))
def dsig(x):
    return np.exp(-x) / ((1 + np.exp(-x))**2)

def MSE(y_p, y):
    return (y - y_p)**2

class NeuralNetwork():

    def __init__(self):
        self.w1 = np.random.normal(size = (2, 4))
        self.b1 = np.random.normal(size = (4))

        self.w2 = np.random.normal(size = (4, 1))
        self.b2 = np.random.normal(size = (1))

    def feedforward(self, x):
        h = Relu(np.dot(x, self.w1) + self.b1)
        print(h)
        return sig(np.dot(h, self.w2) + self.b2)

    def train(self, x_train, y_train):
        for i in range(epochs):
            for x, y in zip(x_train, y_train):
                #feedforward
                l_1 = np.dot(x, self.w1) + self.b1
                h = Relu(l_1)

                l_2 = np.dot(h, self.w2) + self.b2
                o = sig(l_2)

                delta1 = dRelu(l_1[0]) *self.w2[0] * dsig(l_2) * -2*(y - o)
                delta2 = dRelu(l_1[1]) *self.w2[1] * dsig(l_2) * -2*(y - o)
                delta3 = dRelu(l_1[2]) *self.w2[2] * dsig(l_2) * -2*(y - o)
                delta4 = dRelu(l_1[3]) *self.w2[3] * dsig(l_2) * -2*(y - o)
                delta5 =dsig(l_2) * -2*(y - o)

                self.w1[0][0] -= x[0] * lr * delta1
                self.w1[1][0] -= x[1] * lr * delta1

                self.w1[0][1] -= x[0] * lr * delta2
                self.w1[1][1] -= x[1] * lr * delta2

                self.w1[0][2] -= x[0] * lr * delta3
                self.w1[1][2] -= x[1] * lr * delta3

                self.w1[0][3] -= x[0] * lr * delta4
                self.w1[1][3] -= x[1] * lr * delta4

                self.b1[0] -= lr * delta1
                self.b1[1] -= lr * delta2
                self.b1[2] -= lr * delta3
                self.b1[3] -= lr * delta4

                self.w2[0] -= h[0] * lr * delta5
                self.w2[1] -= h[1] * lr * delta5
                self.w2[2] -= h[2] * lr * delta5
                self.w2[3] -= h[3] * lr * delta5

                print((y-o)**2)





nn = NeuralNetwork()
nn.train(x, y)
