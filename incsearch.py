import numpy as np
import matplotlib.pyplot as plt

x1=np.linspace(3, 6, 50)
func1=lambda x1: np.sin(np.dot(10.0,x1))+np.cos(np.dot(3.0, x1))
f1=func1(x1)
plt.figure(1)
plt.plot(x1,f1, 'ro-')
plt.grid()
#plt.show()

x2=np.linspace(3, 6, 100)
func2=lambda x2: np.sin(np.dot(10.0,x2))+np.cos(np.dot(3.0, x2))
f2=func1(x2)
plt.figure(2)
plt.plot(x2,f2, 'bd-')
plt.grid()
#plt.show()

def incsearch(func, xmin, xmax, ns):
    x=np.linspace(xmin, xmax, ns)
    f=func(x)
    nb=0
    xb=[]

    for k in np.arange(np.size(x)-1):
        if np.sign(f[k]) != np.sign(f[k+1]):
            nb=nb+1
            xb.append(x[k])
            xb.append(x[k+1])

    xbt=np.hstack(xb)
    xb=xbt.reshape(nb, 2)

    return nb, xb

xmin=3; xmax=6
func=lambda x: np.sin(np.dot(10.0, x))+np.cos(np.dot(3.0, x))

# check the 50 points
nb, xb=incsearch(func, 3, 6, 50)
print('number of brackets= ', nb)
print('root interval=', xb)

# check the 100 points
nb1, xb1=incsearch(func, 3, 6, 100)
print('number of brackets= ', nb1)
print('root interval=', xb1)


plt.show()
