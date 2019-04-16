import numpy as np
from sympy import *

def newton_raphson(func, dfunc, xr):
   maxit=50
   es=1.0e-5
   iter=0
   while(1):

       x = symbols('x')
       d = Derivative(x**3-9*x**2+24*x-7, x)

       xrold=xr
       xr = np.float(xrold - func(xrold) / d.doit().subs({x: xrold}))

       #xr=np.float(xr-func(xr)/dfunc(xr))
       iter=iter+1
       if xr != 0:
           ea=np.float(np.abs((np.float(xr)-np.float(xrold))/np.float(xr))*100)

       if np.int(ea <= es) | np.int(iter >= maxit):
           root=xr
           fx=func(xr)
           return root, fx, ea, iter

if __name__ == '__main__':
    fp=lambda x: x**3-9*x**2+24*x-7
    dfp=lambda x: 3*x**2-18*x+24
    root, fx, ea, iter=newton_raphson(fp, dfp, 0.1)
    print('root weight= ', root)
    print('f(root weight, should be zero) =', fx)
    print('ea = should be less than 1.0e-4', ea)
    print('iter =', iter)
