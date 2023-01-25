# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:59:32 2023

@author: 0lszowm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import linprog
from scipy.optimize import minimize
from numdifftools import Jacobian


## polecenia 2.1-2.3
def pol2():
    ##wsp.
    f1a = 2
    f1b = 4
    f2a = -1
    f2b = -3
    f3a = -4
    f3b = 2

    def f(x, y):
        return -y

    def f1(x):
        return f1a * x - f1b

    def f2(x):
        return f2a * x - f2b

    def f3(x):
        return f3a * x - f3b

    x1, = solve(f1a - f2a, f1b - f2b)
    x2, = solve(f1a - f3a, f1b - f3b)
    x3, = solve(f2a - f3a, f2b - f3b)

    y1 = f1(x1)
    y2 = f1(x2)
    y3 = f2(x3)

    plt.figure(0)
    plt.plot(x1, y1, 'go', markersize=6)
    plt.plot(x2, y2, 'go', markersize=6)
    plt.plot(x3, y3, 'go', markersize=6)

    plt.fill([x1, x2, x3, x1], [y1, y2, y3, y1], 'grey', alpha=0.6)

    xr = np.linspace(-5, 5, 100)
    y1r = f1(xr)
    y2r = f2(xr)
    y3r = f3(xr)

    plt.plot(xr, y1r, 'k--')
    plt.plot(xr, y2r, 'k--')
    plt.plot(xr, y3r, 'k--')

    plt.xlim(-2, 4)
    plt.ylim(-10, 10)
    plt.show()
    
    print('--------------')
    print('Polecenie 2.3:')
    print('--------------')
    
    c = [0, 1]
    A = [[2, -1], [-1, -1], [-4, -1]]
    b = [4, -3, 2]
    x_bounds = (None, None)
    y_bounds = (None, None)
    res = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds])
    print(f'res.fun \n {res.fun}')
    print(f'res.x \n {res.x}\n')
    

## polecenia 3.1-3.3
def pol3():
    fun = lambda x: x ** 4 - 4 * x ** 3 - 2 * x ** 2 + 12 * x + 9  # funkcja to optymalizacji
    fun_Jac = lambda x: Jacobian(lambda x: fun(x))(x).ravel()  # Jacobian
    bnds = ((None, float('inf')),) * 1  #bounds
    
    print('--------------')
    print('Polecenie 2.3:')
    print('--------------')
    
    # initial guess
    x0 = 2
    res = minimize(fun, x0, bounds=bnds, jac=fun_Jac)
    print("JAC+HESS: optimal value p*", res.fun)
    print("JAC+HESS: optimal var: x = ", res.x)
    
    
    ## wywołania
if __name__ == '__main__':
    pol2()
    pol3()