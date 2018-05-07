import numpy as np


def nat_eq(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
