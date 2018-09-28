import abc

import numpy as np


class Optimizer(abc.ABC):
    """Abstract base class for all optimizers."""

    @abc.abstractmethod
    def get_update(self, gradient):
        """
        Calculate and return the weight update corresponding to the
        passed in gradient.

        Parameters
        ----------
        gradient : np.ndarray

        Returns
        -------
        delta_w : np.ndarray
        """


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer.

    Parameters
    ----------
    learning_rate : float, 0 < learning_rate
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def get_update(self, gradient):
        delta_w = -self.learning_rate * gradient
        return delta_w


class MomentumOptimizer(Optimizer):
    """
    Stochastic gradient descent optimizer with momentum updates. An
    exponential moving average of the gradients is used for the update.

    Parameters
    ----------
    learning_rate : float, 0 < learning_rate
    beta : float, 0 < beta < 1
        The smoothing factor for the exponential moving average of the
        gradients. This effectively defines the number of previous
        gradients to take into account. A lower `beta` discards
        older gradients faster.
    """

    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.momentum = 0

    def get_update(self, gradient):
        self.momentum = self.beta * self.momentum + (1 - self.beta) * gradient
        delta_w = -self.learning_rate * self.momentum
        return delta_w


class RMSprop(Optimizer):
    """
    Root mean square propagation optimizer. Weight updates are
    adjusted by dividing by the square root of the exponential
    moving average of the square of the gradients.

    Parameters
    ----------
    learning_rate : float, 0 < learning_rate
    beta : float 0 < beta < 1
        The smoothing factor for the exponential moving average of the
        square of the gradients. This effectively defines the number of
        previous gradients to take into account. A lower `beta` discards
        older gradients faster.
    """

    def __init__(self, learning_rate=0.001, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = 1e-8
        self.gradient_average = 0

    def get_update(self, gradient):
        self.gradient_average = self.beta * self.gradient_average + (
            1 - self.beta) * np.square(gradient)
        delta_w = -self.learning_rate * gradient / np.sqrt(
            self.gradient_average + self.epsilon)
        return delta_w


class Adam(Optimizer):
    """
    Adaptive moment estimation optimizer. This optimizer combines momentum
    updates with adaptive learning rates a la RMSprop.

    Parameters
    ----------
    learning_rate : float, 0 < learning_rate
    beta_1 : float, 0 < beta_1 < 1
    beta_2 : float, 0 < beta_2 < 1
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-8
        self.momentum = 0
        self.gradient_average = 0

    def get_update(self, gradient):
        self.momentum = self.beta_1 * self.momentum + (
            1 - self.beta_1) * gradient
        self.gradient_average = self.beta_2 * self.gradient_average + (
            1 - self.beta_2) * np.square(gradient)
        delta_w = -self.learning_rate * self.momentum / np.sqrt(
            self.gradient_average + self.epsilon)
        return delta_w
