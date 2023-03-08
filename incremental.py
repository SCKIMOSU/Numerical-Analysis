import numpy as np

def incsearch(func, xmin, xmax):
    x=np.arange(xmin, xmax+1)
    #np.linspace(xmin, xmax, ns)
    f=func(x)
    nb=0
    xb=[]

    for k in np.arange(np.size(x)-1):
        if np.sign(f[k]) != np.sign(f[k+1]):  # k=141
            nb=nb+1
            xb.append(x[k])
            xb.append(x[k+1])

    return nb, xb


g=9.81; cd=0.25; v=36; t=4;
fp=lambda mp:np.sqrt(g*np.array(mp)/cd)*np.tanh(np.sqrt(g*cd/np.array(mp))*t)-v
nb, xb=incsearch(fp, 1, 200)
print('number of brackets= ',nb)
print('root interval=', xb)
