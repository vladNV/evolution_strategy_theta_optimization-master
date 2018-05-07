import numpy as np


def gradient_descent(x, y, alpha=0.0000063):
    theta = np.zeros(x.shape[1])
    y = np.array(y, dtype='float')
    x = np.array(y, dtype='float')
    for i in range(700000):
        gradient = x.T.dot(x * theta - y) / len(x)
        theta = theta - alpha * gradient
    theta = theta.reshape(13, 1)
    return theta
