import numpy as np

def compute_cost(X, y, theta, evol):
    J = np.empty(0)
    for i in range(0, len(theta[0])):
        m = y.size

        predictions = X.dot(theta[evol][i]).flatten()

        sqErrors = (predictions - y) ** 2

        J = np.append(J, np.array((1.0 / (2 * m)) * sqErrors.sum()))

    return J

def compute_costNotEvol(X, y, theta):
    J = np.empty(0)
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y) ** 2

    J = np.append(J, np.array((1.0 / (2 * m)) * sqErrors.sum()))

    return J