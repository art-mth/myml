import numpy as np


def shuffle(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x[idx], y[idx]


def gen_batches(x, y, batch_size):
    n_samples = x.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        yield x[begin:end], y[begin:end]
