import numpy as np
import math
import random
from costFunction import compute_cost as cost
from costFunction import compute_costNotEvol as costNE
from read import X_train, X_test, Y_test, Y_train

n = 3
lmbd = 10
l = 4
evols = 500

numOfParams = X_train.shape[1]

population = np.random.rand(evols, n, numOfParams)
populations = population


def getRandomPoint(parent):
    # print(parent)
    thetaRand = parent + ((np.random.rand(numOfParams) * 2) - 1) * l / numOfParams
    # print(thetaRand)
    summ = 0
    for i in range(0, numOfParams - 1):
        summ += (parent[i] - thetaRand[i]) ** 2
        thetaRand[numOfParams - 1] = - math.sqrt((l ** 2) - summ) + parent[numOfParams - 1]
    return thetaRand


theta = np.zeros((1, n, numOfParams))

for i in range(3):
    theta[0][i] = getRandomPoint(population[0][i])
print(populations)
print(theta[0])
populations = np.vstack((populations, theta))
print(cost(X_train, Y_train, populations, 0))

for i in range(1, evols):
    for k in range(n):
        populations[i][k] = populations[i - 1][k]
        thetaNewEvol = np.zeros((n, lmbd, numOfParams))
        min = costNE(X_train, Y_train, populations[i - 1][k])
        for j in range(lmbd):
            thetaNewEvol[k][j] = getRandomPoint(population[i - 1][k])
            cur = costNE(X_train, Y_train, thetaNewEvol[k][j])
            if cur < min:
                min = cur
                populations[i][k] = thetaNewEvol[k][j]
            print(cost(X_train, Y_train, populations, i - 1))

print((cost(X_test, Y_test, populations, evols - 1)))
