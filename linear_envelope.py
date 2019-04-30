import numpy as np
import matplotlib.pyplot as plt

def linear_enveope(x,y):

    Cost=[]
    Slope=[]
    for i in np.linspace(-3, 5, 80):
        Ct=(sum((i * x - y) ** 2)) / np.size(x)

        Cost.append(Ct)
        Slope.append(i)

    return Cost, Slope

if __name__ == '__main__':
    x=np.array([1,2,3])
    y=np.array([1,2,3])
    Cost, Slope=linear_enveope(x,y)

    plt.plot(Slope, Cost, '*')
    plt.show()
    plt.grid()
