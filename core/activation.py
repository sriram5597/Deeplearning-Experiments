import numpy as np

def sigmoid(val):
    out = 1 /(1 + np.exp(-val))
    return out

def sigmoid_grad(val):
    return sigmoid(val)* (1 - sigmoid(val))

def relu(val):
    return np.maximum(0, val)

def relu_grad(val):
    return np.where(val > 0, 1, 0)

def activation(val, fn):
    if fn == 'relu':
        return relu(val)
    else:
        return sigmoid(val)

def activation_gradient(val, fn):
    if fn == 'relu':
        return relu_grad(val)
    else:
        return sigmoid_grad(val)