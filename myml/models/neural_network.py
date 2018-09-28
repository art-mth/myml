import numpy as np

import myml.utils
import myml.metrics
from myml.nn import losses


class NeuralNetwork:
    """Neural Network model.

    A `NeuralNetwork` instance can perform a variety of different tasks
    depending on the choosen network architecture.
    
    Parameters
    ----------
    optimizer : instance of myml.nn.optimizer.Optimizer
        See myml/nn/optimizers.py for the available optimizers.

    loss : {"square_loss", "cross_entropy_loss"}
        See myml/nn/losses.py for details on the available losses.
    """

    def __init__(self, optimizer, loss):
        self.layers = []
        self.optimizer = optimizer
        self.loss = losses.get_loss(loss)
        self.loss_derivative = losses.get_loss_derivative(loss)

    def add_layer(self, layer):
        """Adds a layer to the model.

        Parameters
        ----------
        layer : instance of myml.nn.layers.Layer
            See myml/nn/layers.py for the available types of layers.
        """

        if self.layers:
            layer.input_shape = self.layers[-1].output_shape
        layer.initialize(optimizer=self.optimizer)
        self.layers.append(layer)

    def fit(self,
            x,
            y,
            batch_size,
            n_epochs,
            metrics=None,
            validation_data=None):
        """Trains the model on the provided training data.

        Parameters
        ----------
        x : numpy ndarray, shape (n_samples, n_features)
            Training input.
            
        y : numpy ndarray, shape (n_samples, n_outputs)
            Training output.

        batch_size : int
            The number of samples used per gradient update.

        n_epochs : int
            The number of epochs to train the model for.

        metrics : list of {"accuracy"}
            A list of metrics that should be evaluated and printed after each
            epoch. See myml/metrics/metrics.py for details on the available 
            metrics.

        validation_data : tuple (x_val, y_val)
            If specified the metrics will be evaluated on this data in 
            addition to the training data.

        Returns
        -------
        history : dict
            Contains the metrics calculated during training.
        """

        history = {"training": []}
        if validation_data is not None:
            history["validation"] = []
        for i in range(n_epochs):
            x, y = myml.utils.shuffle(x, y)
            for x_batch, y_batch in myml.utils.gen_batches(x, y, batch_size):
                self.fit_batch(x_batch, y_batch)
            print("Epoch {0} complete.".format(i + 1))
            history = self._log_metrics(history, metrics, (x, y),
                                        validation_data)
        return history

    def fit_batch(self, x, y):
        """Fit a single batch.

        Parameters
        ----------
        x : numpy ndarray, shape (batch_size, n_features)

        y : numpy ndarray, shape (batch_size, n_outputs)
        """

        y_pred = self._forward_pass(x)
        loss_grad = self.loss_derivative(y, y_pred)
        self._backward_pass(loss_grad)

    def predict(self, x):
        """Predict outputs for the given input.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)

        Returns
        -------
        y : numpy.ndarray, shape (n_samples, n_outputs)
        """

        return self._forward_pass(x)

    def _forward_pass(self, x):
        """
        Perform a forward pass through the network returning the activations 
        of the last layer.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)

        Returns
        -------
        y : numpy.ndarray, shape (n_samples, n_outputs)
        """

        for layer in self.layers:
            x = layer.forward_pass(x)
        return x

    def _backward_pass(self, loss_grad):
        """
        Perform a backward pass through the network updating the networks
        weights accordingly.

        Parameters
        ----------
        loss_grad : numpy.ndarray, shape (n_samples, n_outputs)
        """

        acc_error = loss_grad
        for layer in reversed(self.layers):
            acc_error = layer.backward_pass(acc_error)

    def _log_metrics(self, history, metrics, training_data, validation_data):
        if metrics is not None:
            training_metrics = self._evaluate_metrics(training_data, metrics)
            print("Training metrics: {0}".format(str(training_metrics)))
            history["training"].append(training_metrics)
            if validation_data is not None:
                validation_metrics = self._evaluate_metrics(
                    validation_data, metrics)
                print("Validation metrics: {0}".format(
                    str(validation_metrics)))
                history["validation"].append(validation_metrics)
        return history

    def _evaluate_metrics(self, data, metrics):
        result = {}
        x, y_true = data
        y_pred = self.predict(x)
        for metric_name in metrics:
            metric = getattr(myml.metrics, metric_name)
            metric_score = metric(y_true, y_pred)
            result[metric_name] = metric_score
        return result
