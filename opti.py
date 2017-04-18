import numpy as np

def sech(x):
    return np.cosh(x)**(-1)

def newton(func, dfunc, xr):
    maxit=50
    es=1.0e-5
    iter=0

    while(1):
        xrold=xr
        xr=np.float(xr-func(xr)/dfunc(xr))
        iter=iter+1

        if xr != 0:
            ea=np.float(np.abs((np.float(xr)-np.float(xrold)))/np.float(xr)*100)

        if np.int(ea<=es) | np.int(iter >= maxit):
            break

    root=xr
    fx=func(xr)


    return root, fx, ea, iter

#fp=lambda x: x**3-9*x**2+24*x-7
#dfp=lambda x: 3*x**2-18*x+24
z0=100; v0=55; m=80; c=15; g=9.81;
fp=lambda t: z0+m/c*(v0+m*g/c)*(1-np.exp(-(c/m)*t))-m*g/c*t
dfp=lambda t: v0*np.exp(-(c/m)*t)-m*g/c*(1-np.exp(-(c/m)*t))

#time=  11.610838471061014
#f(root weight, should be zero) = 1.13686837722e-13
#ea = should be less than 1.0e-4 1.3417333744129332e-11
#iter = 6



root, fx, ea, iter=newton(fp, dfp, 5)
print('time= ', root)
print('f(root weight, should be zero) =', fx)
print('ea = should be less than 1.0e-4', ea)
print('iter =', iter)

print('root weight= ', root)
print('f(root weight, should be zero) =', fx)
print('ea = should be less than 1.0e-4', ea)
print('iter =', iter)

#root weight=  0.3313149095222536
#f(root weight, should be zero) = -1.7763568394002505e-15
#ea = should be less than 1.0e-4 4.208588441649749e-06
#iter = 4



'''''''''
g=9.81; cd=0.25; v=36; t=4;

fp=lambda m: np.sqrt(g*m/cd)*np.tanh(np.sqrt(g*cd/m)*t)-36


dfp=lambda m: (1/2)*np.sqrt(g/(m*cd))*np.tanh(np.sqrt(g*cd/m)*t)-g*t/(2*m)*(sech(np.sqrt(g*cd/m)*t))**2


root, fx, ea, iter=newton(fp, dfp, 140)

print('root= ',  root)
print('fx= ',fx)
print('ea= ',ea)
print('iter= ',iter)

'''''''''


