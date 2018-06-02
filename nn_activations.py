import numpy as np


def sigmoid(Z):
    A = 1/(1+np.exp(-Z))

    return A, Z


def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ

def cross_entropy_cost(AL, Y):
    m = Y.shape[1]
    cost = -1 / m * np.sum(np.sum(Y * (np.log(AL)) + (1 - Y) * (np.log(1 - AL))))
    return np.squeeze(cost)
