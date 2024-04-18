import numpy as np
import matplotlib.pyplot as plt

#
#root interval= [[3.30150754 3.31658291]
 #[3.57286432 3.5879397 ]
 #[3.93467337 3.94974874]
 #[4.2361809  4.25125628]
 #[4.52261307 4.53768844]
 #[4.88442211 4.89949749]
 #[5.17085427 5.18592965]
 #[5.47236181 5.48743719]
 #[5.83417085 5.84924623]]

def dfunc_na(func, xr):
    delta_x=0.0001
    na_func=(func(xr+delta_x)-func(xr))/delta_x
    return na_func

def incsearch(func, xmin, xmax, ns):
    x=np.linspace(xmin, xmax, ns)

    nb=0;     xb=[]
    for k in np.arange(np.size(x)-1):

        if np.sign(dfunc_na(func, x[k])) != np.sign(dfunc_na(func, x[k+1])):
            nb=nb+1
            xb.append(x[k])
            xb.append(x[k+1])

    xbt=np.hstack(xb)
    xb=xbt.reshape(nb, 2)
    return nb, xb

def draw(xmin, xmax, inc, func):
    x = np.linspace(xmin, xmax, inc)
    #func = lambda x: np.sin(np.dot(10.0, x)) + np.cos(np.dot(3.0, x))
    f1 = func(x)
    plt.figure(1)
    plt.plot(x, f1, 'ro-')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    xmin=3; xmax=6
    inc=200
    func=lambda x: np.sin(np.dot(10.0, x))+np.cos(np.dot(3.0, x))
    nb, xb=incsearch(func, xmin, xmax, inc)
    draw(xmin, xmax, inc, func)
    print('number of brackets= ', nb)
    print('root interval=', xb)

