import numpy as np
import pandas as pd

from myml.preprocessing import to_categorical, train_test_split
from myml.models import NeuralNetwork
from myml.nn.optimizers import RMSprop
from myml.nn.layers import Dense
from myml.metrics import accuracy


def main():
    # Prepare the data
    data = pd.read_csv("train.csv")
    X = data.loc[:, "pixel0":].values
    y = data.loc[:, "label"].values
    Y = to_categorical(y, num_categories=10)
    X_train, X_holdout, Y_train, Y_holdout = train_test_split(
        X, Y, test_size=0.2)

    training_mean = np.mean(X_train, axis=0)
    X_train = (X_train - training_mean) / 255
    X_holdout = (X_holdout - training_mean) / 255

    # Build the model
    optimizer = RMSprop()
    nn = NeuralNetwork(optimizer=optimizer, loss="cross_entropy_loss")
    nn.add_layer(Dense(128, activation="relu", input_shape=(784, )))
    nn.add_layer(Dense(10, activation="softmax"))

    # Train the model
    history = nn.fit(
        X_train,
        Y_train,
        batch_size=128,
        n_epochs=20,
        metrics=["accuracy"],
        validation_data=(X_holdout, Y_holdout))

    validation_accuracy = history["validation"][-1]["accuracy"]
    print("Validation accuracy: {0}".format(validation_accuracy))


if __name__ == '__main__':
    main()
