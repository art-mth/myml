import numpy as np


def to_categorical(labels, num_categories):
    result = np.zeros((len(labels), num_categories))
    for i, label in enumerate(labels):
        result[i, label] = 1.0
    return result


def train_test_split(x, y, test_size):
    shuffled_indices = np.random.permutation(len(x))
    test_set_size = int(len(x) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return (x[train_indices], x[test_indices], y[train_indices],
            y[test_indices])
