# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:52:10 2023

@author: 0lszowm
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from numpy.linalg import matrix_rank


## polecenia 2.1-2.5
def pol2(time=None):
    if time is None:
        time = 10
    #  wektor czasu
    t = np.linspace(0, time, time*100+1)

    def input_signal(t):
        return np.heaviside(t, 1)

    def model(z0, t):
        dz1dt = z0[0]*np.log(z0[1])
        dz2dt = - z0[1]*np.log(z0[0]) + z0[1]*input_signal(t)
        return [dz1dt, dz2dt]
    z0 = [1, 1]
    z = odeint(model, z0, t)
    plt.figure("polecenie 2.1-2.3")
    plt.plot(t, z[:, 0], label='z0')
    plt.plot(t, z[:, 1], label='z1')
    plt.xlabel('t[s]')
    plt.legend(loc='best')
    plt.grid()
    plt.title('nieliniowy system wykres')
    plt.show()

    def model_new(x0, t):
        dx1dt = x0[1]
        dx2dt = -x0[0]+input_signal(t)
        return [dx1dt, dx2dt]

    x0 = [np.log(z0[0]), np.log(z0[1])]
    x = odeint(model_new, x0, t)
    plt.figure("polecenie 2.5")
    plt.plot(t, x[:, 0], label='x0')
    plt.plot(t, x[:, 1], label='x1')
    plt.plot(t, np.exp(x[:, 0]), label='z0')
    plt.plot(t, np.exp(x[:, 1]), label='z1')
    plt.xlabel('t[s]')
    plt.legend(loc='best')
    plt.grid()
    plt.title('nowe + przeliczone zmienne wykres')
    plt.show()
    

## polecenia 5.1-5.6
def pol5(time=None, u=None, wspolne_wykr=None, osobne_wykr=None):
    if wspolne_wykr is None:
        wspolne_wykr = True
    if osobne_wykr is None:
        osobne_wykr = False

    if u is None:
        u=1

    def wykresy_osobne(t, x_nielin, x_lin):
        plt.figure(2)
        plt.plot(t, x_nielin[:, 0], label='x0')
        plt.plot(t, x_nielin[:, 1], label='x1')
        plt.xlabel('t[s]')
        plt.legend(loc='best')
        plt.grid()
        plt.title('nieliniowy system wykres')
        plt.show()
        plt.figure(3)
        plt.plot(t, x_lin[:, 0], label='x0')
        plt.plot(t, x_lin[:, 1], label='x1')
        plt.xlabel('t[s]')
        plt.legend(loc='best')
        plt.grid()
        plt.title('liniowy system wykres')
        plt.show()

    def wykresy_wspolne(t, x_nielin, x_lin):
        plt.figure(4)
        plt.plot(t, x_nielin[:, 0], label='model nieliniowy')
        plt.plot(t, x_lin[:, 0], "r--", label='model liniowy')
        plt.xlabel('t[s]')
        plt.legend(loc='best')
        plt.grid()
        plt.title('zmienna stanu x0')
        plt.show()
        plt.figure(5)
        plt.plot(t, x_nielin[:, 1], label='model nieliniowy')
        plt.plot(t, x_lin[:, 1], "r--", label='model liniowy')
        plt.xlabel('t[s]')
        plt.legend(loc='best')
        plt.grid()
        plt.title('zmienna stanu x1')
        plt.show()

    def sterowalnosc(A, B):
        arr = []
        for i in range(A.shape[0]):
            # print((A**i)@B)
            arr.append((A ** i) @ B)
        ret = np.concatenate(arr, axis=1)
        if A.shape[0] == matrix_rank(ret):
            print('Uklad jest sterowalny')
        else:
            print('Uklad jest nie-sterowalny')

    if time is None:
        time = 10
    #  wektor czasu
    t = np.linspace(0, time, time*1000+1)
    R = 1
    m = 9
    J = 1
    g = 10
    d = 0.5

    def input_signal(t):
        return np.array([[u*np.heaviside(t, 1)]])

    def model(x0, t):
        dx1dt = x0[1]
        dx2dt = 1/J * input_signal(t) - d/J*x0[1] - (m*g)/J*R*np.sin(x0[0])
        return [dx1dt, dx2dt[0, 0]]

    x0 = [0, 0]
    x_nielin = odeint(model, x0, t)
    A = np.array([[0, 1],
                  [-(m*g*R)/J, -d/J]])
    B = np.array([[0],
                  [1/J]])

    def model_lin(x_we, t):
        x = np.array([[x_we[0]],
                      [x_we[1]]])
        dxdt = A@x+B@input_signal(t)
        dxdt = np.squeeze(dxdt)
        return dxdt

    x0 = [0, 0]
    x_lin = odeint(model_lin, x0, t)

    if wspolne_wykr is True:
        wykresy_wspolne(t, x_nielin, x_lin)
    if osobne_wykr is True:
        wykresy_osobne(t, x_nielin, x_lin)

    sterowalnosc(A, B)

    A = np.array([[0, 1],
                  [(-(m * g * R) / J) * (np.sqrt(2)/2), -d / J]])
    B = np.array([[0],
                  [1 / J]])

    def input_signal_ulep(t):
        return np.array([[u * np.heaviside(t, 1)-(45*np.sqrt(2))*np.heaviside(t, 1)]])

    def model_lin_ulep(x_we, t, x0):
        x = np.array([[x_we[0]-x0[0]],
                      [x_we[1]-x0[1]]])
        dxdt = A@x+B@input_signal_ulep(t)
        dxdt = np.squeeze(dxdt)
        return dxdt

    x0_ulep = [-np.pi/4, 0]
    x_lin_ulep = odeint(model_lin_ulep, x0_ulep, t,args=(x0_ulep,))
    x_lin_ulep = np.subtract(x_lin_ulep, x0_ulep)
    plt.figure(5)
    plt.plot(t, x_nielin[:, 0], label='model nieliniowy')
    plt.plot(t, x_lin_ulep[:, 0], "r--", label='model liniowy')
    plt.xlabel('t[s]')
    plt.legend(loc='best')
    plt.grid()
    plt.title('zmienna stanu x0')
    plt.show()
    plt.figure(99)
    plt.plot(t, x_nielin[:, 1], label='model nieliniowy')
    plt.plot(t, x_lin_ulep[:, 1], "r--", label='model liniowy')
    plt.xlabel('t[s]')
    plt.legend(loc='best')
    plt.grid()
    plt.title('zmienna stanu x1')
    plt.show()
    
    
## wywołania
if __name__ == '__main__':
    #pol2()
    pol5(time=10, u=45*np.sqrt(2), wspolne_wykr=False, osobne_wykr=True)
    
    