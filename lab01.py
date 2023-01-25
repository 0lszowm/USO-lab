"""
Created on Wed Jan 25 14:05:35 2023

@author: 0lszowm
"""

import numpy as np
from numpy.linalg import matrix_rank as rank
from numpy.linalg import inv
import matplotlib.pyplot as plt


## polecenie 2.1
def pol21():
    print('--------------')
    print('Polecenie 2.1:')
    print('--------------')
    x1 = 3 ** 12 - 5
    print(f'x = {x1}\n')
    x2 = np.array([[2, 0.5]]) @ np.array([[1, 4], [-1, 3]]) @ np.array([[-1], [-3]])
    print(f'x = {x2}\n')
    x3 = np.array([[1, -2, 0], [-2, 4, 0], [2, -1, 7]])
    print(f'x = {rank(x3)}\n')
    x4 = inv(np.array([[1, 2], [-1, 0]])) @ np.array([[-1], [2]])
    print(f'x = {x4}\n')


## polecenie 2.2
def pol22():
    print('--------------')
    print('Polecenie 2.2:')
    print('--------------')
    tab = np.array([1, 1, -129, 171, 1620])
    x1 = -46
    x2 = 14
    y1 = np.polyval(tab, x1)
    y2 = np.polyval(tab, x2)
    print(f'Wartość wielomianu w punkcie x={x1} wynosi: {y1}\n')
    print(f'Wartość wielomianu w punkcie x={x2} wynosi: {y2}\n')


## polecenia 3.1-3.2
def pol3():
    print('--------------')
    print('Polecenia 3.1-3.2:')
    print('--------------')
    tab = np.array([1, 1, -129, 171, 1620])
    x1 = -46 ## dolna granica przedziału
    x2 = 14 ## górna granica przedziału
    precision = 1
    n = int((x2-x1)/precision)+1
    range = np.linspace(-46, 14, n)
    results = []
    for x in range:
        results.append(np.polyval(tab, x))
    print(f'najwyższa wartość wielomianu w zakresie <{x1};{x2}> wynosi: {max(results)}\n')
    print(f'najniższa wartość wielomianu w zakresie <{x1};{x2}> wynosi: {min(results)}\n')


## polecenia 4.1-4.2
def pol4(coeffs, lower_limit, upper_limit, precision):
    n = int((upper_limit-lower_limit)/precision)+1
    range = np.linspace(lower_limit,upper_limit, n)
    results = []
    for x in range:
        results.append([np.polyval(coeffs, x)])
    return [max(results), min(results)]


## polecenia 5.1-5.2
def pol5(coeffs, lower_limit, upper_limit, precision):
    n = int((upper_limit-lower_limit)/precision)+1
    range = np.linspace(lower_limit,upper_limit, n)
    results = []
    for x in range:
        results.append([np.polyval(coeffs, x)])
    plt.figure("Wykres z poleceń 5.1-5.2 ")
    plt.plot(range, results, label="y=f(x)")
    plt.legend(loc="upper right")
    plt.title(f"Wykres funkcji o współczynnikach {coeffs}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()


## wywołania
if __name__ == '__main__':
    pol21()
    pol22()
    pol3()
    pol4(coeffs=[3,-1,-1,0,2], lower_limit=-10, upper_limit=10, precision=0.1)
    pol5(coeffs=[3,-1,-1,0,2], lower_limit=-10, upper_limit=10, precision=0.1)
