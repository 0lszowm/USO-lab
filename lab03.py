# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:28:33 2023

@author: 0lszowm
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
import sys


## polecenia 1.1-1.3
def pol1(uklad=None, time=None):
    if uklad is None:
        uklad = 1;
    if time is None:
        time =10
    match uklad:
        ## polecenie 1.1
        case 1:
            A = np.matrix([[-1/2,0],
                           [1/4,-1/2]])
            B = np.matrix([[1/2],
                           [1/4]])
            C = np.matrix([1,0])
            D = np.matrix([0])
        case 2:
            A = np.matrix([[-1,0,0],
                           [0,-1/2,0],
                           [0,1/6,-1/3]])
            B = np.matrix([[1],
                           [1/2],
                           [1/6]])
            C = np.matrix([1,0,0])
            D = np.matrix([0])
        case 3:
            A = np.matrix([-1/2])
            B = np.matrix([0])
            C = np.matrix([1])
            D = np.matrix([0])
        case 4:
            A = np.matrix([[-4,0,-2],
                           [0,0,1],
                           [1/2,-1/2,-1/2]])
            B = np.matrix([[2],
                           [0],
                           [0]])
            C = np.matrix([1,0,0])
            D = np.matrix([0])
        case _:
            print("nieprawidlowy numer ukladu")
            sys.exit(42)
    ## polecenie 1.2
    arr = []
    for i in range(A.shape[0]):
        arr.append((A**i)@B)
    ret = np.concatenate(arr, axis=1)
    if matrix_rank(ret) != A.shape[0]:
        print(f'Uklad nr {uklad} jest nie-sterowalny')
    else:
        print(f'Uklad nr {uklad} jest sterowalny')
    ## polecenie 1.3
    t = np.linspace(0, time, time*100+1)
    sys = signal.StateSpace(A, B, C, D)
    T, y_out, x_out = signal.lsim(system=sys, U=np.heaviside(t, 1), T=t)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(T, y_out, label="Y(t)")
    ax1.grid()
    ax1.legend(loc="upper right")
    ax1.set_title(f"Odpowiedz ukladu nr {uklad} na wymuszenie skokowe")
    
    if(np.ndim(x_out)>1):
        for i in range(np.size(x_out[0])):
            ax2.plot(T, x_out[:,i], label=f"x{i}(t)")
    else:
        ax2.plot(T, x_out, label=f"x1(t)")
    
    ax2.grid()
    ax2.legend(loc="upper right")
    ax2.set_title(f"Przebieg zmiennych stanu ukladu nr {uklad} przy wymuszeniu skokowym")
    
    T, y_out, x_out = signal.lsim(system=sys, U=np.sin(t), T=t)
    ax3.plot(T, y_out, label="Y(t)")
    ax3.legend(loc="upper right")
    ax3.grid()
    ax3.legend(loc="upper right")
    ax3.set_title(f"Odpowiedz ukladu nr {uklad} na wymuszenie sinusoidalne")
    
    if(np.ndim(x_out)>1):
        for i in range(np.size(x_out[0])):
            ax4.plot(T, x_out[:,i], label=f"x{i}(t)")
    else:
        ax4.plot(T, x_out, label=f"x1(t)")
    ax4.grid()
    ax4.legend(loc="upper right")
    ax4.set_title(f"Przebieg zmiennych stanu ukladu nr {uklad} przy wymuszeniu sinusoidalnym")

        
        
## wywołania
if __name__ == '__main__':
    pol1(uklad=1)
    