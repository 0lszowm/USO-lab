# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:16:29 2023

@author: 0lszowm
"""

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import sympy as sp


## polecenia 2.1-2.5
def pol1(time=None):
    if time is None:
        time = 5
    t = np.linspace(0, time, time * 1000 + 1)
    A = np.array([[0, 1],
                  [0, 0]])
    B = np.array([[0],
                  [1]])

    def model(x_in, t):
        x = np.array(x_in)
        u = input_signal(t)
        dxdt = A @ x + B @ u
        dxdt = np.squeeze(dxdt).astype(float)
        return dxdt

    def calculate_la_x(T):
        a_1 = sp.Symbol('a1')
        a_2 = sp.Symbol('a2')
        t = sp.Symbol('t')
        C1 = sp.Symbol('C1')
        C2 = sp.Symbol('C2')
        la1 = a_1
        la2 = -a_1 * t + a_2
        u = -la2
        x2 = sp.integrate(u, t) + C1
        x1 = sp.integrate(x2, t) + C2
        #  print(f'x1\n {x1}')
        #  print(f'x2\n {x2}')
        x2_0 = x2.subs(t, 0)
        c1 = sp.solvers.solve(x2_0 - 1, C1, dict=True)[0][C1]
        x1_0 = x1.subs(t, 0)
        x1_0 = x1_0.subs(C1, c1)
        c2 = sp.solvers.solve(x1_0 - 1, C2, dict=True)[0][C2]
        x1_1 = x1.subs(t, T)
        x1_1 = x1_1.subs(C1, c1)
        x1_1 = x1_1.subs(C2, c2)
        a1 = sp.solvers.solve(x1_1, a_1, dict=True)[0][a_1]
        x2_1 = x2.subs(t, T)
        x2_1 = x2_1.subs(C1, c1)
        x2_1 = x2_1.subs(C2, c2)
        x2_1 = x2_1.subs(a_1, a1)
        a2 = sp.solvers.solve(x2_1, a_2, dict=True)[0][a_2]
        a1 = a1.subs(a_2, a2)
        #  print(f'a1 {a1}')
        #  print(f'a2 {a2}')
        la1 = la1.subs(a_1, a1)
        la2 = la2.subs(a_1, a1)
        la2 = la2.subs(a_2, a2)
        #  print(f'la1 {la1}')
        #  print(f'la2 {la2}')
        x1 = x1.subs(C1, c1)
        x1 = x1.subs(C2, c2)
        x1 = x1.subs(a_1, a1)
        x1 = x1.subs(a_2, a2)
        #  print(f'x1 {x1}')
        x2 = x2.subs(C1, c1)
        x2 = x2.subs(a_1, a1)
        x2 = x2.subs(a_2, a2)
        #  print(f'x2 {x2}')
        return {'la1': la1, 'la2': la2, 'x1': x1, 'x2': x2}

    return_dict = calculate_la_x(time)

    def la_1(t_in):
        return return_dict['la1'].subs('t', t_in)

    def la_2(t_in):
        return return_dict['la2'].subs('t', t_in)

    def x_1(t_in):
        return return_dict['x1'].subs('t', t_in)

    def x_2(t_in):
        return return_dict['x2'].subs('t', t_in)

    def input_signal(t):
        return np.array([-la_2(t)])

    def x_12(t):
        x1 = []
        x2 = []
        for t_o in t:
            x1.append(x_1(t_o))
            x2.append(x_2(t_o))
        return x1, x2

    x0 = [1, 1]
    sol = odeint(model, x0, t)
    plt.figure(0)
    plt.plot(t, sol)
    plt.grid()
    plt.show()

    x1, x2 = x_12(t)
    plt.figure(1)
    plt.plot(t, x1)
    plt.plot(t, x2)
    plt.grid()
    plt.show()
    
## wywołania
if __name__ == '__main__':
    pol1(time=10) 
