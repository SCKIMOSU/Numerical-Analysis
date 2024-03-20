import numpy as np


def bisect(func, xl, xu):
    maxit = 100;
    es = 1.0e-4
    test = func(xl) * func(xu)

    if test > 0:
        print('No sign change')
        return [], [], [], []

    iter = 0;
    xr = xl;
    ea = 100

    while (1):
        xrold = xr
        xr = float((xl + xu) / 2)
        iter = iter + 1

        if xr != 0:  # 나누기에서 분모가 0이면 안 되죠. 0으로 나누는 것은 ZeroDivisionError: division by zero 가 발생하죠
            ea = float(np.abs(float(xr) - float(xrold)) / float(xr)) * 100

        test = func(xl) * func(xr)

        if test > 0:
            xl = xr
        elif test < 0:
            xu = xr
        else:
            ea = 0

        if int(ea < es) | int(iter >= maxit):
            break

    root = xr
    fx = func(xr)

    return root, fx, ea, iter


if __name__ == '__main__':
    fm = lambda m: np.sqrt(9.81 * m / 0.25) * np.tanh(np.sqrt(9.81 * 0.25 / m) * 4) - 36
    root, fx, ea, iter = bisect(fm, 40, 200)
    print('root = ', root, '(Bisection)')
    print('f(root) = ', fx, '(must be zero, Bisection)')
    print('estimated error= ', ea, '(must be zero error, Bisection)')
    print('iterated number to find root =', iter, '(Bisection)')
