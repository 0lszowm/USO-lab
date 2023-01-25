# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:22:20 2023

@author: 0lszowm
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


## polecenia 2.1-2.2
def pol20(time=None):
    if time is None:
        time=5
    # parametry
    r1 = 2
    r2 = 5
    c1 = 0.5
    l1 = 2
    l2 = 0.5
    A = np.array([[-r2/l2, 0, -1/l2],
                  [0, -r1/l1, 1/l1],
                  [1/c1, -1/c1, 0]])
    B = np.array([[0],
                  [1/l1],
                  [0]])
    C = np.array([1, 0, 0])
    D = 0

    def model(x, t):
        x_np = np.array(x)
        dxdt = A@x_np
        dxdt = dxdt.reshape(dxdt.size, 1) + B * np.heaviside(t, 0)
        dxdt = dxdt.reshape(1, dxdt.size)

        return dxdt[0]


    # wywolania
    t = np.linspace(0, time, time*100+1)
    u = [0, 0, 0]
    x = odeint(model, u, t)
    # print(x)
    y = C@x.T
    plt.figure("polecenie 2.2")
    plt.plot(t, y)
    plt.title("przebieg sygnału wyjsciowego przy wymuszeniu skokowym")
    plt.grid()
    plt.show()
    

## polecenie 2.3-2.4
def pol21(time=None):
    if time is None:
        time = 10
    r1 = 2
    r2 = 5
    c1 = 0.5
    l1 = 2
    l2 = 0.5
    A = np.array([[-r2/l2, 0, -1/l2],
                  [0, -r1/l1, 1/l1],
                  [1/c1, -1/c1, 0]])
    B = np.array([[0],
                  [1/l1],
                  [0]])
    C = np.array([1, 0, 0])
    D = 0
    # task value
    yd = 3
    kp = 3
    ki = 1.2
    kd = 0.1


    def model(x, t):
        x_np = np.array(x[0:3])
        err_c = np.array(x[3:6])
        err = yd-x_np
        x_np = x_np-kp*err-ki*err_c
        # print(x_np)
        dxdt = A@x_np
        dxdt = dxdt.reshape(dxdt.size, 1) + B * np.heaviside(1, 0)
        dxdt = dxdt.reshape(1, dxdt.size)
        #print(err)
        dxdt = dxdt[0]
        dxdt = np.append(dxdt, err)
        # print(dxdt)
        return dxdt


    # wywolania
    t = np.linspace(0, time, time*1000+1)
    u = [0, 0, 0, 0, 0, 0]
    x = odeint(model, u, t, args=())
    #print(x)
    x = x[:, 0:3]
    y = C@x.T
    plt.figure("polecenie 2.4")
    plt.plot(t, y)
    plt.title(f"Przebieg odpowiedzi układu zamknietego \n"
              f"dla nastaw kp = {kp}, ki = {ki}, kd = {kd}")
    plt.grid()
    plt.show()
    
    
## wywołania
if __name__ == '__main__':
    pol20()
    pol21()