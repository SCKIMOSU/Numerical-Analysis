import numpy as np
import matplotlib.pyplot as plt

def incsearch(func, xmin, xmax):
    x=np.arange(xmin, xmax+1)
    #np.linspace(xmin, xmax, ns)
    f=func(x)
    nb=0
    xb=[]

    for k in np.arange(np.size(x)-1):
        if np.sign(f[k]) != np.sign(f[k+1]):
            nb=nb+1
            xb.append(x[k])
            xb.append(x[k+1])

    return nb, xb

def draw():
    x = np.linspace(40, 200, 50)
    fp = lambda mp: np.sqrt(g * np.array(mp) / cd) * np.tanh(np.sqrt(g * cd / np.array(mp)) * t) - v
    #func = lambda x: np.sin(np.dot(10.0, x)) + np.cos(np.dot(3.0, x))
    f1 = fp(x)
    plt.figure(1)
    plt.plot(x, f1, 'ro-')
    plt.grid()


if __name__ == '__main__':
    g=9.81; cd=0.25; v=36; t=4
    fp=lambda mp:np.sqrt(g*np.array(mp)/cd)*np.tanh(np.sqrt(g*cd/np.array(mp))*t)-v
    nb, xb=incsearch(fp, 40, 200)
    print('number of brackets= ',nb)
    print('root interval=', xb)
