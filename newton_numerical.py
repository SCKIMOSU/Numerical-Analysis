import numpy as np
from scipy.optimize import fsolve

def sech(x):
    return np.cosh(x) ** (-1)

def dfunc_na(func, xr):
    delta_x=0.0001
    na_func=(func(xr+delta_x)-func(xr))/delta_x
    return na_func

def newton_raphson(func, dfunc, xr, xt):
    maxit = 50
    es = 1.0e-5
    iter = 0

    while (1):
        xrold = xr
        xr = float(xr - func(xr) / dfunc(xr))

        xr_na = float(xr - func(xr) / dfunc_na(func, xr))

        iter = iter + 1

        if xr != 0:
            ea = float(np.abs((float(xr) - float(xrold)) / float(xr)) * 100)
            et= float(np.abs((float(xt) - float(xr)) / float(xt)) * 100)

        if int(ea <= es) | int(iter >= maxit):
            break

        root = xr
        root_na = xr_na
        fx = func(xr)
        fx_na = func(root_na)
    return root, root_na, fx, fx_na, ea, iter

if __name__ == '__main__':
    g = 9.81; cd = 0.25; v = 36; t = 4
    fm = lambda m: np.sqrt(9.81 * m / 0.25) * np.tanh(np.sqrt(9.81 * 0.25 / m) * 4) - 36
    xt = fsolve(fm, 1)
    print("Real Root= ", xt)

    fp = lambda m: np.sqrt(g * m / cd) * np.tanh(np.sqrt(g * cd / m) * t) - v
    dfp = lambda m: (1 / 2) * np.sqrt(g / (m * cd)) * np.tanh(np.sqrt(g * cd / m) * t) - g * t / (2 * m) * (
    sech(np.sqrt(g * cd / m) * t)) ** 2

    root, root_na, fx, fx_na, ea, iter = newton_raphson(fp, dfp, 140, xt)
    print('root weight= ', root)
    print('root by numerical derivative= ', root_na)
    print('f(root weight, should be zero) =', fx)
    print('f(root by numerical derivative, should be zero) =', fx_na)
    print('ea = should be less than 1.0e-4', ea)
    print('iter =', iter)
