import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as ig

x=np.arange(0, 0.8, 0.01)
f= lambda x: 0.2+25*x-200*x**2+675*x**3-900*x**4+400*x**5

sol = ig.quad(f, 0, 0.8)
re=np.real(sol)
real=re[0]

plt.figure(1)
plt.plot(x, f(x), 'ro-')

#  n=1 segment
I=(0.8-0)*(f(0.8)+f(0))/2
# 0.1728000000000225

trap=np.ones(80)*(f(0.8)+f(0))/2

plt.plot(x, trap, 'b*-')

error=(real-I)/real*100

#  n=2 segments
#
I1=(0.8-0)*(f(0)+2*f(0.4)+f(0.8))/(2*2)

trap1=np.ones(40)*(f(0.4)+f(0))/2

plt.plot(x[0:39], trap1, 'c*-')

trap2=np.ones(40)*(f(0.4)+f(0.8))/2
plt.plot(x[40:80], trap2, 'c*-')

# I1=1.0688000000000115

error1=(real-I1)/real*100

# 34.850455136540162 %

# n=4 segments
I2=(0.8-0)*(f(0)+2*(f(0.2)+f(0.4)+f(0.6))+f(0.8))/(4*2)
# I2= 1.484800000000001

error2=(real-I2)/real*100
# error2= 9.4928478543561035
