import abc
import copy

import numpy as np

from myml.nn import activations


class Layer(abc.ABC):
    """Abstract base class for all layers."""

    @property
    @abc.abstractmethod
    def output_shape(self):
        pass

    @abc.abstractmethod
    def forward_pass(self, x):
        """
        Perform a forward pass through the layer returning the output
        activations.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, *input_shape)

        Returns
        -------
        y : numpy.ndarray, shape (n_samples, *output_shape)
        """

    @abc.abstractmethod
    def backward_pass(self, acc_error):
        """
        Perform a backward pass through the layer returning the accumulated
        error. This method also updates the layers weights accordingly.

        Parameters
        ----------
        acc_error : numpy.ndarray, shape (n_samples, *output_shape)

        Returns
        -------
        acc_error : numpy.ndarray, shape (n_samples, *input_shape)
        """


class Dense(Layer):
    """Fully connected neural network layer.

    Parameters
    ----------
    n_units : int
        The number of units in the layer.

    activation : {"identity", "sigmoid", "tanh", "softmax", "relu", "leaky_relu"}
        See myml/nn/activations.py for details on the available activations.

    input_shape : tuple, (input_dim_1, ..., input_dim_n)
        The dimensions of the input examples. The `input_shape` only needs to
        be specified for the first layer of a network.

    Attributes
    ----------
    weights : numpy.ndarray, shape (*input_shape, *output_shape)

    biases : numpy.ndarray, shape (*output_shape, )

    input_activations : numpy.ndarray (n_samples, *input_shape)

    weighted_inputs : numpy.ndarray (n_samples, *output_shape)
    """

    def __init__(self, n_units, activation, input_shape=None):
        self.n_units = n_units
        self.input_shape = input_shape
        self.weights = None
        self.biases = None
        self.input_activations = None
        self.weighted_inputs = None
        self.batch_size = None
        self.activation = activations.get_activation(activation)
        self.activation_derivative = activations.get_activation_derivative(
            activation)

    @property
    def output_shape(self):
        return (self.n_units, )

    def initialize(self, optimizer):
        self.weights = np.random.randn(*self.input_shape,
                                       *self.output_shape) * np.sqrt(
                                           2 / self.input_shape[0])
        self.biases = np.zeros(self.output_shape)
        self.weights_opt = copy.deepcopy(optimizer)
        self.biases_opt = copy.deepcopy(optimizer)

    def forward_pass(self, x):
        self.batch_size = x.shape[0]
        self.input_activations = x
        self.weighted_inputs = np.dot(x, self.weights) + self.biases
        output_activations = self.activation(self.weighted_inputs)
        return output_activations

    def backward_pass(self, acc_error):
        error = acc_error * self.activation_derivative(self.weighted_inputs)
        acc_error_updated = np.dot(error, self.weights.T)
        nabla_w = np.dot(self.input_activations.T, error) / self.batch_size
        nabla_b = np.mean(error, axis=0)
        self.weights += self.weights_opt.get_update(nabla_w)
        self.biases += self.biases_opt.get_update(nabla_b)
        return acc_error_updated
