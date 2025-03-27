import numpy as np
import matplotlib.pyplot as plt

def falseposition(func, xl, xu):
    maxit = 100
    es = 1.0e-4

    test = func(xl) * func(xu)

    if test > 0:
        print('no sign change')
        return None, None, None, None

    iter = 0
    xr = xl
    ea = 100.0

    while True:
        xrold = xr
        fl = func(xl)
        fu = func(xu)

        denominator = fl - fu
        if denominator == 0:
            print("Division by zero in false position formula.")
            break

        xr = xu - fu * (xl - xu) / denominator
        iter += 1

        if xr != 0:
            ea = abs((xr - xrold) / xr * 100)

        test = func(xl) * func(xr)

        if test > 0:
            xl = xr
        elif test < 0:
            xu = xr
        else:
            ea = 0.0

        if ea <= es or iter >= maxit:
            break

    root = xr
    fx = func(xr)

    return root, fx, ea, iter

if __name__ == '__main__':
    fm = lambda m: np.sqrt(9.81 * m / 0.25) * np.tanh(np.sqrt(9.81 * 0.25 / m) * 4) - 36
    root, fx, ea, iter = falseposition(fm, 40, 200)

    print('root =', root)
    print('f(root) =', fx)
    print('ea =', ea)
    print('iter =', iter)
