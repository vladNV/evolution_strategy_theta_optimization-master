import numpy as np


def rmse(real, pred):
    real = np.array(real)
    pred = np.array(pred)
    return np.sqrt(((pred - real) ** 2).mean())
