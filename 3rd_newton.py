import numpy as np
import matplotlib.pyplot as plt
x1=np.linspace(0,10,50)
x2=np.linspace(-10,10,50)
y1=x1**3-9*x1**2+24*x1-7
y2=x2**3-9*x2**2+24*x2-7
plt.figure(1)
plt.plot(x1, y1, 'ro-')
plt.grid()
plt.show()
plt.figure(2)
plt.plot(x2, y2, 'b*-')
plt.grid()
plt.show()

import numpy as np
from scipy.optimize import fsolve
fx=lambda x: x**3-9*x**2+24*x-7
x=fsolve(fx, 1)
print("Real Root= ", x)






def newton_raphson(func, dfunc, xr):
   maxit=50
   es=1.0e-5
   iter=0
   while(1):
       xrold=xr
       xr=np.float(xr-func(xr)/dfunc(xr))
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

    p = [3, -18, 24]
    xd1 = np.roots(p)
    print('xd1 =', xd1) 




