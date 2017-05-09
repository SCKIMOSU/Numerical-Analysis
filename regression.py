import numpy as np
import matplotlib.pyplot as plt

x=np.array([0,1,2,3,4,5])
y=np.array([0, 0.8, 0.9, 0.1, -0.8, -1])

n=np.size(x)

b=(n*np.sum(x*y)-(np.sum(x)*np.sum(y)))/(n*np.sum(x**2)-(np.sum(x))**2)
# b= -0.30285714285714288

a=(np.sum(y)-b*np.sum(x))/n
# a= 0.75714285714285723

fx=b*x+a

p1=np.polyfit(x, y, 1)
# p1=array([-0.30285714,  0.75714286])
p2=np.polyfit(x, y, 2)
# array([-0.16071429,  0.50071429,  0.22142857])
p3=np.polyfit(x, y, 3)
# array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254])


plt.figure(1)
plt.plot(x,y, 'o')
plt.plot(x,fx, 'r*-')

plt.figure(2)
plt.plot(x,y, 'o')
plt.plot(x, np.polyval(p1, x), 'b*-')
# = plt.plot(x,fx, 'r*-')


plt.figure(3)
plt.plot(x,y, 'o')
plt.plot(x, np.polyval(p1,x), 'r*-')
plt.plot(x, np.polyval(p2,x), 'b>-')
plt.plot(x, np.polyval(p3,x), 'mx-')





