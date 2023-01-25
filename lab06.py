# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:36:22 2023

@author: 0lszowm
"""

import numpy as np
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


## polecenia 2.1-2.6
def pol2(time=None, x0=None):
    if time is None:
        time = 5
    if x0 is None:
        x0 = [0, 0]
    R = 0.5
    C = 0.5
    L = 0.2
    A = np.array([[0, 1],
                  [-1 / (L * C), -R / L]])
    B = np.array([[0],
                  [1 / L]])
    Q_matrix = np.eye(2)
    R_matrix = np.eye(1)
    P_matrix = solve_continuous_are(a=A, b=B, q=Q_matrix, r=R_matrix)
    K_matrix = R_matrix ** (-1) * B.T @ P_matrix
    
    print('--------------')
    print('Polecenie 2.1:')
    print('--------------')
    print(f"Wzmocnienia K są rowne: {K_matrix}")

    def model(x, t):
        x_np = np.array(x[0:2])
        dxdt = A @ x_np
        U = -K_matrix @ x_np
        dxdt = dxdt + B @ U
        J = x_np.T @ Q_matrix @ x_np + U.T * R @ U
        dxdt = np.append(dxdt, J)
        return dxdt

    t = np.linspace(0, time, time * 100 + 1)
    u = np.append(x0, 0)  # last element to wskaznik jakosci
    x = odeint(model, u, t)
    plt.figure("Polecenie 2.5")
    plt.plot(t, x[:, 0], label='x_0')
    plt.plot(t, x[:, 1], label='x_1')
    plt.plot(t, x[:, 2], label='J')
    plt.legend(loc='best')
    plt.title(f'symulacja układu dla warunków początkowych x0 = {x0}')
    plt.grid()
    plt.show()
    print(f"Ostateczny wskaznik jakosci wynosi {x[-1, 2]}")
    
    
## polecenia 3.1-3.8
def pol3(time=None):
    if time is None:
        time = 5
    R = 0.5
    C = 0.5
    L = 0.2
    A = np.array([[0, 1],
                  [-1 / (L * C), -R / L]])
    B = np.array([[0],
                  [1 / L]])
    Q_matrix = np.eye(2)
    R_matrix = np.eye(1)
    S_matrix = np.array([[1, 1],
                  [1, 1]])
    S = S_matrix.reshape(4,)
    t_ric = np.linspace(time, 0, time * 100 + 1)
    t = np.linspace(0, time, time*100+1)

    def riccati(p, t):
        if p.shape != (2, 2):
            P = p.reshape(2, 2)
        else:
            P = p
        dpdt = np.array(-1 * (P @ A - P @ B @ R_matrix ** (-1) @ B.T @ P + A.T @ P + Q_matrix))
        dpdt = dpdt.reshape(4, )
        return dpdt

    def model_skok(x, t):
        x_np = np.array(x)
        dxdt = A @ x_np
        dxdt = dxdt.reshape(dxdt.size, 1) + B * np.heaviside(1, 0)
        dxdt = dxdt.reshape(1, dxdt.size)

        return dxdt[0]

    def model(x, t, f_0, f_1, f_2, f_3):
        P_matrix = np.array([[f_0(t), f_1(t)],
                             [f_2(t), f_3(t)]])
        K_matrix = R_matrix ** (-1) * B.T @ P_matrix
        x_np = np.array(x[0:2])
        u = -K_matrix @ x_np
        J = x_np.T @ Q_matrix @ x_np + u.T@R_matrix@u  # wskaznik jakosci
        dxdt = A @ x_np
        dxdt = dxdt.reshape(dxdt.size, 1) + B * u
        dxdt = dxdt.reshape(1, dxdt.size)
        dxdt = dxdt[0]
        dxdt = np.append(dxdt, J)

        return dxdt

    P = odeint(riccati, S, t_ric)
    plt.figure(0)
    plt.plot(t_ric, P)
    plt.grid()
    plt.title(f'Przebieg elemetów macierzy P w czasie (t0={0};t1={time})')
    plt.show()
    f0 = interp1d(t_ric, P[:, 0], fill_value='extrapolate')
    f1 = interp1d(t_ric, P[:, 1], fill_value='extrapolate')
    f2 = interp1d(t_ric, P[:, 2], fill_value='extrapolate')
    f3 = interp1d(t_ric, P[:, 3], fill_value='extrapolate')
    #####
    x0 = [0, 0]  # warunki poczatkowe
    #####
    y = odeint(model_skok, x0, t)
    plt.figure(1)
    plt.plot(t, y)
    plt.grid()
    plt.title(f'odpowiedz obiektu na wymuszenie skokowe w czasie (t0={0}; t1={time}) \n'
              f'dla warunków początkowych x0= {x0}')
    plt.show()
    #####
    x0 = [0.5, 0, 0]  # warunki poczatkowe, last element wskaznik jakosci
    #####
    y = odeint(model, x0, t, args=(f0, f1, f2, f3))
    last = y[-1:][0]
    print(last)
    J = abs(last[0:2].T@S_matrix@last[0:2])+last[2]  # Wskaznik jakosci koncowy
    print(f'Koncowy wskaznik jakosci J={J}')
    plt.figure(2)
    plt.plot(t, y[:, 0], label='x_0')
    plt.plot(t, y[:, 1], label='x_1')
    plt.plot(t, y[:, 2], label='J')
    plt.legend(loc='best')
    plt.grid()
    plt.title('Regulator LQR skonczony horyzont')
    plt.show()
    
    
## polecenia 4.1-4.5
def pol4(time=5, qd=0, x0=None):
    if x0 is None:
        x0 = [0, 0]
    R = 0.5
    C = 0.5
    L = 0.2
    A = np.array([[0, 1],
                  [-1 / (L * C), -R / L]])
    B = np.array([[0],
                  [1 / L]])
    q_d = qd
    xd = np.array([q_d, 0])
    Q_matrix = np.eye(2)
    R_matrix = np.eye(1)
    P_matrix = solve_continuous_are(a=A, b=B, q=Q_matrix, r=R_matrix)
    K_matrix = R_matrix ** (-1) * B.T @ P_matrix
    print(K_matrix)

    def model(x, t):
        x_np = np.array(x[0:2])
        e = xd - x_np
        dxdt = A @ x_np
        U = q_d / C + K_matrix @ e
        dxdt = dxdt + B @ U
        J = x_np.T @ Q_matrix @ x_np + U.T * R @ U
        dxdt = np.append(dxdt, J)
        return dxdt

    t = np.linspace(0, time, time * 100 + 1)
    u = np.append(x0, 0)  # last element to wskaznik jakosci
    x = odeint(model, u, t)
    plt.figure("polecenie 4.5")
    plt.plot(t, x[:, 0], label='x_0')
    plt.plot(t, x[:, 1], label='x_1')
    plt.plot(t, x[:, 2], label='J')
    plt.legend(loc='best')
    plt.title(f'symulacja układu zamknietego dla wartosci zadanej qd = {q_d} \n dla warunków początkowych x0 = {x0}')
    plt.grid()
    plt.show()


## wywołania
if __name__ == '__main__':
    pol2(time=5, x0=[0.1, 0.5])
    pol3(time=2)
    pol4(time=2, qd=1, x0=[0.1, 0.5])
    