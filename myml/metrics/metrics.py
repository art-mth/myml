import numpy as np


def accuracy(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy_score = np.sum(np.equal(y_true, y_pred)) / len(y_true)
    return accuracy_score
