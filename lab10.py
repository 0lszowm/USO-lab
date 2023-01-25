# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 19:53:22 2023

@author: 0lszowm
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are


m = 9
J = 1
d = 0.5
g = 9.81
l=1
 
def a_matrix(x):
    x1, x2 = x
    return np.array([[0, 1],
                      [m*g*l*np.sin(x1)/x1, -d/J]])

def b_matrix(x):
    x1, x2 = x
    return np.array([[0],
                      [1/J]])

def input_signal(t):
    return np.array([0])


## polecenia 1.1-1.2
def pol1(time=None):
    if time is None:
        time = 10
    def model(x_in, t):
        x1, x2 = x_in
        u = input_signal(t)
        dxdt1 = x2
        dxdt2 = -d/J*x2+m*g*l*np.sin(x1)+1/J*u
        return [dxdt1, dxdt2[-1]]
    ## polecenie 1.1
    t = np.linspace(0, time, time * 100 + 1)
    x = [np.pi/4, 0]
    sol = odeint(model, x, t)
    plt.figure("polecenie 1.1")
    plt.plot(t, sol[:, 0], label="x0")
    plt.plot(t, sol[:, 1], label="x1")
    plt.grid()
    plt.legend(loc="best")
    plt.title('Symulacja układu wachadła')
    plt.show()
    
    ## polecenie 1.2
    def model1(x_in, t):
        x = np.array(x_in)
        u = input_signal(t)
        A = a_matrix(x)
        B = b_matrix(x)
        dxdt = A @ x + B @ u
        return dxdt
    x = [np.pi/4, 0]
    sol = odeint(model1, x, t)
    plt.figure("polecenie 1.2")
    plt.plot(t, sol[:, 0], label="x0")
    plt.plot(t, sol[:, 1], label="x1")
    plt.grid()
    plt.title('Symulacja układu wachadła z parametryzacja')
    plt.legend(loc="best")
    plt.show()
    
## polecenia 2.1-2.3
def pol2(time=None):
    if time is None:
        time=10
    t = np.linspace(0, time, time*100+1)
    Q_matrix = np.eye(2)
    
    def k_matrix(r, b, a, q):
        p = solve_continuous_are(a=a, b=b, q=q, r=r)
        return r**-1*b.T@p
    
    def R_matrix(x):
        return np.eye(1)

    def model(x_in, t):
        x = np.array(x_in)
        A = a_matrix(x)
        B = b_matrix(x)
        K = k_matrix(r=R_matrix(x), a=A, b=B, q=Q_matrix)
        u = -K @ x
        dxdt = A @ x + B @ u
        return dxdt
    
    x = [np.pi / 4, 0]
    sol = odeint(model, x, t)
    plt.figure("polecenie 2.1")
    plt.plot(t, sol[:, 0], label="x0")
    plt.plot(t, sol[:, 1], label="x1")
    plt.legend()
    plt.grid()
    plt.title('polecenie 2.1')
    plt.show()
    
    ## polecenie 2.2
    x = [2*np.pi, 0]
    sol = odeint(model, x, t)
    plt.figure("polecenie 2.2")
    plt.plot(t, sol[:, 0], label="x0")
    plt.plot(t, sol[:, 1], label="x1")
    plt.legend()
    plt.grid()
    plt.title('polecenie 2.2')
    plt.show()
    
    ## polecenie 2.3
    def Q_matrix1(x):
        x1, x2 = x
        return np.array([[x1**2, 0],
                         [0, x2**2]])
    def model1(x_in, t):
        x = np.array(x_in)
        A = a_matrix(x)
        B = b_matrix(x)
        K = k_matrix(r=R_matrix(x), a=A, b=B, q=Q_matrix1(x))
        u = -K @ x
        dxdt = A @ x + B @ u
        return dxdt
    x = [3, 0]
    sol = odeint(model, x, t)
    plt.figure("polecenie 2.3")
    plt.plot(t, sol[:, 0], label="x0")
    plt.plot(t, sol[:, 1], label="x1")
    plt.legend()
    plt.grid()
    plt.title('polecenie 2.3')
    plt.show()
    

## wywołania
if __name__ == '__main__':
    pol1()
    pol2(time=10)