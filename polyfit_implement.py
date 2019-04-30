import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10, 90, 10.)
y = np.array([25, 70, 380, 550, 610, 1220, 830, 1450])

plt.figure(1)
plt.plot(x, y, 'ro-')
plt.grid()

xsum=np.sum(x)
ysum=np.sum(y)
xysum=sum(x*y)
n=np.size(x)
xavg=xsum/n
yavg=ysum/n

a1=(n*xysum-xsum*ysum)/(n*sum(x**2)-xsum**2)

a0= yavg-xavg*a1

plt.figure(2)
y1=a1*x+a0
plt.plot(x, y, 'ro-', x, y1, 'b*-')
plt.grid()

p1=np.polyfit(x,y,1)
# array([  19.4702381 , -234.28571429])

plt.figure(3)
y1=a1*x+a0
plt.plot(x, y, 'ro-', x, y1, 'b*-', x, np.polyval(p1, x), 'mp-')
plt.grid()
