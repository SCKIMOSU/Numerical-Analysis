import numpy as np
import matplotlib.pyplot as plt

def incsearch(func, xmin, xmax, ns):
    x=np.linspace(xmin, xmax, ns)
    f=func(x)
    nb=0;     xb=[]
    for k in np.arange(np.size(x)-1):
        if np.sign(f[k]) != np.sign(f[k+1]):
            nb=nb+1
            xb.append(x[k])
            xb.append(x[k+1])

    xbt=np.hstack(xb)
    xb=xbt.reshape(nb, 2)
    return nb, xb

def draw():
    x = np.linspace(3, 6, 50)
    func = lambda x: np.sin(np.dot(10.0, x)) + np.cos(np.dot(3.0, x))
    f1 = func(x)
    plt.figure(1)
    plt.plot(x, f1, 'ro-')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    xmin=3; xmax=6
    func=lambda x: np.sin(np.dot(10.0, x))+np.cos(np.dot(3.0, x))
    nb, xb=incsearch(func, 3, 6, 50)
    draw()
    print('number of brackets= ', nb)
    print('root interval=', xb)
