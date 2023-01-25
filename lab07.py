# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:40:13 2023

@author: 0lszowm
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint


## polecenia 2.1-2.4
def pol2(time=10):
    def model(c, t):
        ddt = t ** 2
        return ddt

    t = np.linspace(0, time, time * 100 + 1)
    c = 0
    result = odeint(model, c, t)
    analytical_result = 1 / 3 * time ** 3 + c
    print(f'Numeryczne rozwiązanie równania wynosi y={float(result[-1, :])}')
    print(f'Analityczne rozwiązanie równania wynosi y={analytical_result}')
    plt.figure("Polecenie 2.4")
    plt.plot(t, result[:, 0])
    plt.xlabel('t[s]')
    plt.grid()
    plt.title('Równanie nieliniowe pierwszego rzędu')
    plt.show()


## polecenia 3.1-3.5
def pol3(time=50):
    kp = 2
    to_w = 4
    dziwne_c = 0.25

    def u(t):
        return np.heaviside(t, 0)

    def model(x, t):
        y = x[0]
        dydt = x[1]
        dy2dt2 = -(2 * dziwne_c / to_w) * dydt - (1 / to_w) * np.sqrt(y) + (kp / to_w ** 2) * u(t)
        return [dydt, dy2dt2]

    t = np.linspace(0, time, time * 100 + 1)
    x0 = [0, 0]  # warunki poczatkowe
    result = odeint(model, x0, t)
    # print(result)
    plt.figure("Polecenie 3.5")
    plt.plot(t, result[:, 0], label='x1')
    plt.plot(t, result[:, 1], label='x2')
    plt.legend(loc='best')
    plt.xlabel('t[s]')
    plt.grid()
    plt.title('Równanie nieliniowe drugiego rzędu')
    plt.show()
    

## polecenia 4.1-4.5
def pol4(time=10, a=1, limitations=True):
    kp = 2
    T = 2
    k_ob = 4
    upper_limit = 0.1
    lower_limit = -0.1
    t = np.linspace(0, time, time * 100 + 1)

    def x(t, a):
        return a * np.heaviside(t, 1)

    def feedback(y, t, a=1):
        u = kp * (x(t, a) - y)
        u_limited = np.clip(u, lower_limit, upper_limit)
        if limitations:
            return (k_ob * u_limited - y) / T  # bo czlon inercyjny 1-rzedu
        else:
            return (k_ob * u - y) / T  # bo czlon inercyjny 1-rzedu

    y0 = 0
    result = odeint(feedback, y0, t, args=(a,))
    plt.figure("polecenie 4.5")
    plt.plot(t, result)
    plt.xlabel('t[s]')
    plt.grid()
    if limitations:
        plt.title(f'Układ z ograniczeniami dla sygnału sterujacego u(t)={a}*1(t)')
    else:
        plt.title(f'Układ bez ograniczen dla sygnału sterujacego u(t)={a}*1(t)')
    plt.show()
    
    
## wywołania
if __name__ == '__main__':
    pol2()
    pol3()
    pol4(time=15, a=2, limitations=True)