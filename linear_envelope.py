import numpy as np
import matplotlib.pyplot as plt

def linear_enveope(x,y):
    Cost=[]
    Slope=[]
    for i in np.arange(-3, 5, 0.1): #np.linspace(-3, 5, 80):
        Ct=(sum((i * x - y) ** 2)) / np.size(x)
        Cost.append(Ct)
        Slope.append(i)
    return Cost, Slope

def draw(Cost,Slope):
    plt.plot(Slope, Cost, '*')
    plt.show()
    plt.grid()
    plt.xlabel('Slope')
    plt.ylabel('Cost')


if __name__ == '__main__':
    x=np.array([1,2,3])
    y=np.array([1,2,3])
    Cost, Slope=linear_enveope(x,y)
    zipped=list(zip(Slope, Cost))
    draw(Cost, Slope)
