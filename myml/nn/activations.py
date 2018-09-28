import numpy as np


def identity(z):
    return z


def identity_derivative(z):
    return np.ones(z.shape)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    return 2 / (1 + np.exp(-2 * z)) - 1


def tanh_derivative(z):
    return 1 - np.square(tanh(z))


def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)


def softmax_derivative(z):
    s = softmax(z)
    return s * (1 - s)


def relu(z):
    return np.where(z >= 0, z, 0)


def relu_derivative(z):
    return np.where(z >= 0, 1, 0)


def leaky_relu(z):
    np.where(z >= 0, z, z * 0.01)


def leaky_relu_derivative(z):
    return np.where(z >= 0, 1, 0.01)


_activations = {
    "identity": identity,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "relu": relu,
    "leaky_relu": leaky_relu,
}

_activation_derivatives = {
    "identity": identity_derivative,
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "softmax": softmax_derivative,
    "relu": relu_derivative,
    "leaky_relu": leaky_relu_derivative,
}


def get_activation(identifier):
    return _activations[identifier]


def get_activation_derivative(identifier):
    return _activation_derivatives[identifier]
