import numpy as np
import pandas as pd
from read import X_train, X_test, Y_train, Y_test
import time
from nat_eq import nat_eq
from rmse import rmse
import matplotlib.pyplot as plt
from grad_dec import gradient_descent
from genetic_optimization import genetic as gen
import warnings

warnings.simplefilter("ignore")


if __name__ == '__main__':
    # time0_nat = time.time()
    # time1_nat = time.time() - time0_nat
    # theta_nat = nat_eq(X_train, Y_train)
    # Y_pred_nat = np.dot(X_test, theta_nat)
    # loss_nat = rmse(Y_pred_nat, Y_test)
    # print()
    # print('Theta, normal equation:')
    # print(theta_nat)
    # print('Time to find solution with normal equation: ', time1_nat)
    # print('Normal equation RMSE: ', loss_nat)
    # print()
    # plt.plot(Y_pred_nat, Y_test, 'bo')
    # plt.show()
    #
    # time0_gd = time.time()
    # time1_gd = time.time() - time0_gd
    # theta_gd = gradient_descent(X_train, Y_train)
    # Y_pred_gd = np.dot(X_test, theta_gd)
    # loss_gd = rmse(Y_pred_gd, Y_test)
    # print('Theta, gradient descent:')
    # print(theta_gd)
    # print('Time to find solution with gradient descent: ', time1_gd)
    # print('Gradient descent RMSE: ', loss_gd)
    # print()
    # plt.plot(Y_pred_gd, Y_test, 'bo')
    # plt.show()

    time0_gen = time.time()
    time1_gen = time.time() - time0_gen
    theta_gen = gen(X_train, Y_train)
    Y_pred_gen = np.dot(X_test, theta_gen)
    loss_gd = rmse(Y_pred_gen, Y_test)
    print('Theta, evolution:')
    print(theta_gen)
    print('Time to find solution with evolution strategy: ', time1_gen)
    print('Evolution strategy RMSE: ', loss_gd)
    print()
    plt.plot(Y_pred_gen, Y_test, 'bo')
    plt.show()

