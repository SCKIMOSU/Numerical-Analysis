import numpy as np
import matplotlib.pyplot as plt

def newton_euler(func, xl):
    maxit=100
    es=1.0e-5

    iter=0
    xr=xl

    ea=100

    while(1):
        xrold=xr
        #xr=np.float((xl+xu)/2)
        #xr = np.float(xu-func(xu)*(xl-xu)/(func(xl)-func(xu)))
        den=func(xrold+1)-func(xrold)
        xr=xrold-func(xrold)/den

        iter=iter+1

        if xr != 0:
            ea=np.float(np.abs(np.float(xr)-np.float(xrold))/np.float(xr))*100

        if np.int(ea<=es) | np.int(iter>=maxit):
            break
    root=xr
    fx=func(xr)


    return root, fx, ea, iter
#root= 142.73765563964844
#      142.73763311

if __name__ == '__main__':

    fm=lambda m: np.sqrt(9.81*m/0.25)*np.tanh(np.sqrt(9.81*0.25/m)*4)-36
    root, fx, ea, iter=newton_euler(fm, 40)

    print('root=', root)
    print('f(root)=', fx)
    print('ea=', ea)
    print('iter=', iter)


'''''''''false position
root= 142.73783844758216
f(root)= 4.20034974979e-06
ea= 7.781013789779357e-05
iter= 29
'''''''''


''''''''' bisection
root= 142.73765563964844
f(root)= 4.60891321552e-07
ea= 5.3450468252827136e-05
iter= 21
'''''''''
