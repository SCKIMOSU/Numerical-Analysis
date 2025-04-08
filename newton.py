import numpy as np
from scipy.optimize import fsolve

def sech(x):
    return 1 / np.cosh(x)

def newton_raphson(func, dfunc, xr, xt):
    maxit = 50
    es = 1.0e-5
    iter = 0

    while True:
        xrold = xr
        xr = float(xr - func(xr) / dfunc(xr))
        iter += 1

        if xr != 0:
            ea = abs((xr - xrold) / xr) * 100
            et = abs((xt - xr) / xt) * 100

        if ea <= es or iter >= maxit:
            break

    root = xr
    fx = func(xr)
    return root, fx, ea, iter

if __name__ == '__main__':
    g = 9.81
    cd = 0.25
    v = 36
    t = 4

    # 방정식 정의
    fm = lambda m: np.sqrt(g * m / cd) * np.tanh(np.sqrt(g * cd / m) * t) - v

    # fsolve로 실제 해 추정
    xt = fsolve(fm, 1)[0]
    print("Real Root =", xt)

    # 뉴턴-랩슨용 함수 및 도함수
    fp = lambda m: np.sqrt(g * m / cd) * np.tanh(np.sqrt(g * cd / m) * t) - v
    dfp = lambda m: (0.5) * np.sqrt(g / (m * cd)) * np.tanh(np.sqrt(g * cd / m) * t) \
                    - (g * t / (2 * m)) * (sech(np.sqrt(g * cd / m) * t)) ** 2

    root, fx, ea, iter = newton_raphson(fp, dfp, 140, xt)
    print('Root weight =', root)
    print('f(root weight, should be zero) =', fx)
    print('ea (should be < 1.0e-4) =', ea)
    print('iterations =', iter)
