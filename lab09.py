# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 19:27:03 2023

@author: 0lszowm
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are


## polecenia 2.1-2.5
def pol2(time=None):
    if time is None:
        time = 5
    l = 1
    m = 9
    J = 1
    d = 0.5
    g = 9.81

    def input_signal(t):
        return np.heaviside(t, 1)*0

    def model(x, t):
        dxdt1 = x[1]
        dxdt2 = input_signal(t)/J+d/J*x[1]-(m*g*l)/J*np.sin(x[0])
        return [dxdt1, dxdt2]

    t = np.linspace(0, time, time*100+1)

    x0 = [np.pi/4, 0]
    sol = odeint(model, x0, t)
    plt.figure("polecenie 2.2")
    plt.plot(t, sol[:, 0], label="x0")
    plt.plot(t, sol[:, 1], label="x1")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

    x0_lin = [np.pi, 0]
    u0_lin = 0

    A = np.array([[0, 1],
                  [m*g*l/J, -d/J]])
    B = np.array([[0],
                  [1/J]])
    Q_matrix = np.eye(2)
    R_matrix = np.eye(1)
    P_matrix = solve_continuous_are(a=A, b=B, q=Q_matrix, r=R_matrix)
    K_matrix = R_matrix ** (-1) * B.T @ P_matrix

    def input_signal_lin_k_odeint(x, t):
        x_daszek = x - x0_lin
        return -K_matrix@x_daszek+u0_lin

    x0_lin2 = [np.pi-0.1, 0]

    def model_lin_k_odeint(x, t):
        dxdt1 = x[1]
        dxdt2 = input_signal_lin_k_odeint(x, t) / J + d / J * x[1] - (m * g * l) / J * np.sin(x[0])
        return [dxdt1, dxdt2[-1]]
    solv = odeint(model_lin_k_odeint, x0_lin2, t)
    plt.figure("polecenie 2.5-2.6")
    plt.plot(t, solv[:, 0], label="x0")
    plt.plot(t, solv[:, 1], label="x1")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    pol2(time=5)
