# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:39:10 2023

@author: 0lszowm
"""

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import matplotlib.pyplot as plt


## polecenia 2.1-2.5
def pol2():
    ## polecenie 2.2
    kp = 3
    T = 2
    A = -1 / T
    B = kp / T
    C = 1
    D = 0
    ## polecenie 2.3
    num = [kp]
    den = [T, 1]
    eq = signal.TransferFunction(num, den)
    ## polecenie 2.4
    t,y = signal.step(eq)
    plt.figure("polecenie 2.4")
    plt.plot(t, y)
    plt.xlabel('Czas [s]')
    plt.ylabel('Wartość')
    plt.title('Odpowiedz skokowa obiektu na podstawie transmitancji ze skryptu')
    plt.grid()
    plt.show()
    ## polecenie 2.5
    eq_s = signal.StateSpace(A, B, C, D)
    t, y = signal.step(eq_s)
    plt.figure("polecenie 2.5")
    plt.plot(t, y)
    plt.xlabel('Czas [s]')
    plt.ylabel('Wartość')
    plt.title('Odpowiedz skokowa obiektu na podstawie równania stanu ze skryptu')
    plt.grid()
    plt.show()
    ## polecenie 2.6
    def model(t,y):
        return -t/T+kp/T

    ## polecenie 2.7-2.10
    t = np.linspace(0, 15, 151)
    u = np.heaviside(t, 0)
    y = odeint(model, u, t)
    plt.figure("polecenie 2.9")
    plt.plot(t, y[:, 0])
    plt.title("Odpowiedź skokowa obiektu na podstawie \n bezposredniego rozwiązania równania różniczkowego")
    plt.xlabel('Czas [s]')
    plt.ylabel('Wartość')
    plt.grid()
    plt.show()
    

## polecenia 3.1-3.4
def pol3():
    ## parametry ukladu
    R = 12
    L = 1
    c = 0.0001
    
    ## polecenie 3.1
    
    ## transmitancja
    num = [1, 0]
    den = [L, R, 1/c]
    sys = signal.TransferFunction(num, den)
    print(f'tf result: {sys}')
    ## odpowiedz skokowa
    t, y = signal.step(sys)
    plt.figure("polecenie 3.1")
    plt.plot(t, y, label="odpowiedz skokowa")
    ## odpowiedz impulsowa
    t, y = signal.impulse(sys)
    plt.plot(t, y, label="odpowiedz impulsowa")
    plt.xlabel('Czas [s]')
    plt.ylabel('Wartość')
    plt.title('Odpowiedz ukladu RLC na podstawie transmitancji operatorowej')
    plt.legend(loc="upper right")
    plt.show()
    
    ## polecenie 3.2
    
    ## Zmienne stanu
    A = np.array([[0,1],[-1/(L*c),-R/L]])
    B = np.array([[0],[1]])
    C = np.array([0,1])
    D = 0
    sys = signal.StateSpace(A, B, C, D)
    print(f'ss result: {sys}')
    ## odpowiedz skokowa
    t, y = signal.step(sys)
    plt.figure("polecenie 3.2")
    plt.plot(t, y, label="odpowiedz skokowa")
    ## odpowiedz impulsowa
    t, y = signal.impulse(sys)
    plt.plot(t, y, label="odpowiedz impulsowa")
    plt.xlabel('Czas [s]')
    plt.ylabel('Wartość')
    plt.title('Odpowiedz ukladu RLC na podstawie zmiennych stanu')
    plt.legend(loc="upper right")
    plt.show()

    sstf = signal.ss2tf(A, B, C, D)
    print(f'ss2tf result: {sstf}')

    tfss = signal.tf2ss(num, den)
    print(f'tf2ss result: {tfss}')
    

## polecenia 4.1-4.4
def pol4():
    ## parametry ukladu
    m = 1
    L = 0.5
    d = 0.1
    J = 1/3*m*L**2

    ## zmienne stanu
    A = np.array([[0, 1], [0, -d/J]])
    B = np.array([[0], [1/J]])
    C = np.array([1, 0])
    D = 0
    sys = signal.StateSpace(A, B, C, D)
    
    ## polecenie 4.2
    ## odpowiedz skokowa
    t, y = signal.step(sys)
    plt.figure("polecenie 4.2")
    plt.plot(t, y)
    plt.xlabel('Czas [s]')
    plt.ylabel('Wartość')
    plt.title('Odpowiedz skokowa ukladu manipulatora na podstawie zmiennych stanu')
    plt.grid()
    plt.show()
    
    ## polecenie 4.3
    t = np.linspace(0, 10, 1001)
    ## odpowiedz na sygnal we liniowo narastajacy
    u = np.linspace(0, 1, 1001)
    t_out, y_out, x_out = signal.lsim2(sys, u, t)
    plt.figure("polecenie 4.3")
    plt.plot(t_out, y_out, label="sygnał liniowo narastający")
    plt.title('Odpowiedz ukladu manipulatora na sygnal liniowo narastajacy')
    ## odpowiedz na sygnal we liniowo opadajacy
    u = np.linspace(1, 0, 1001)
    t_out, y_out, x_out = signal.lsim2(sys, u, t)
    plt.plot(t_out, y_out, label="sygnał liniowo opadający")
    plt.xlabel('Czas [s]')
    plt.ylabel('Wartość')
    plt.title('Odpowiedz ukladu manipulatora na:')
    plt.grid()
    plt.legend(loc="upper right")
    plt.show()
    
    ## polecenie 4.4
    ## charakterystyka bodego
    w, mag, phase = signal.bode(sys)
    plt.figure("polecenie 4.4-1")
    plt.semilogx(w, mag)  # Bode magnitude plot
    plt.title('Charakterystyka amplitudowa bodego')
    plt.grid()
    plt.show()
    plt.figure("polecenie 4.4-2")
    plt.semilogx(w, phase)  # Bode phase plot
    plt.title('Charakterystyka fazowa bodego')
    plt.grid()
    plt.show()
    

## wywołania
if __name__ == '__main__':
    pol2()
    pol3()
    pol4()