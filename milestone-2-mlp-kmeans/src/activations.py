import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        return 1 /(1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        # This cacluation is currently expensive, we could implement a cache, 
        # or save the result in mlp fit from feed_forward, and then pass as argument to the gradient function in back_prop
        s = Sigmoid.forward(z) 

        # The formula below is the derivative of sigmoid function, written in a more elegant way
        # see for ex: https://www.geeksforgeeks.org/machine-learning/derivative-of-the-sigmoid-function/
        return s * (1 - s) # max value is 0.25 when z = 0, so be careful about gradient vanishing pb

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0.0, z)

    @staticmethod
    def gradient(z):
        return np.where(z > 0.0, 1.0, 0.0)

class Identity:
    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def gradient(z):
        return np.ones_like(z)


class Softmax:
    @staticmethod
    def forward(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def gradient(z):
        return np.ones_like(z)
