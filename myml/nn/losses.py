import numpy as np


def square_loss(y, y_pred):
    return 0.5 * np.square(y_pred - y)


def square_loss_derivative(y, y_pred):
    return y_pred - y


def cross_entropy_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)


def cross_entropy_loss_derivative(y, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -y / y_pred + (1 - y) / (1 - y_pred)


_losses = {
    "square_loss": square_loss,
    "cross_entropy_loss": cross_entropy_loss,
}

_loss_derivatives = {
    "square_loss": square_loss_derivative,
    "cross_entropy_loss": cross_entropy_loss_derivative,
}


def get_loss(identifier):
    return _losses[identifier]


def get_loss_derivative(identifier):
    return _loss_derivatives[identifier]
