import numpy as np

def bisect(func, xl, xu):
    maxit=100
    es=1.0e-4

    test=func(xl)*func(xu)

    if test > 0:
        print('No sign change')
        return [], [], [], []

    iter=0
    xr=xl

    ea=100

    while (1):
        xrold=xr
        xr=np.float((xl+xu)/2)

        iter=iter+1

        if xr != 0:
            ea=np.float(np.abs(np.float(xr)-np.float(xrold))/np.float(xr))*100

        test=func(xl)*func(xr)

        if test > 0:
            xl=xr
        elif test < 0:
            xu=xr
        else:
            ea=0

        if np.int(ea<es) | np.int(iter >= maxit):
            break

    root=xr
    fx=func(xr)

    return root, fx, ea, iter

if __name__ == '__main__':
    
    fm=lambda m: np.sqrt(9.81*m/0.25)*np.tanh(np.sqrt(9.81*0.25/m)*4)-36
    root, fx, ea, iter=bisect(fm, 40, 200)
    print('root = ', root, '(Bisection)')
    print('f(root) = ', fx, '(must be zero, Bisection)')
    print('estimated error= ', ea, '(must be zero error, Bisection)')
    print('iterated number to find root =', iter, '(Bisection)')
