import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as ig

def simpson(f, a, b, n):
    h=(b-a)/n
    k=0.0
    x=a + h

    for i in np.arange(1, n/2+1):
        k += 4*f(x)
        x += 2*h
        x = a + 2*h

    for i in np.arange(1,n/2):
        k += 2*f(x)
        x += 2*h

    return (h/3)*(f(a)+f(b)+k)


if __name__ == '__main__':
    a=0
    b=0.8
    n=7
    #x = np.arange(a, b, 0.01)
    f = lambda x: 0.2 + 25 * x - 200 * x ** 2 + 675 * x ** 3 - 900 * x ** 4 + 400 * x ** 5
    simpson=simpson(f, a, b, n)
    print("simpson=", simpson)
    sol = ig.quad(f, a, b)
    re = np.real(sol)
    real = re[0]
    print("real=", real)
    error = (real - simpson) / real * 100
    print("error=", error)
    
