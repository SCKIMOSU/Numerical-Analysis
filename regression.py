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
p4=np.polyfit(x, y, 4) # new in 2020/4/22


plt.figure(1)
plt.plot(x,y, 'o')
plt.plot(x,fx, 'r*-')

plt.figure(2)
plt.plot(x,y, 'o')
plt.plot(x, np.polyval(p1, x), 'b*-')
# = plt.plot(x,fx, 'r*-')
plt.show()


plt.figure(3)
plt.plot(x,y, 'o')
plt.plot(x, np.polyval(p1,x), 'r*-')
plt.plot(x, np.polyval(p2,x), 'b>-')
plt.plot(x, np.polyval(p3,x), 'mx-')
plt.plot(x, np.polyval(p4,x), 'go-')
plt.show()

plt.figure(4)
plt.plot(x, y, 'o')
plt.grid()
plt.show()
xp=np.linspace(-2, 6, 100)

plt.plot(xp, np.polyval(p1,xp), 'r-')
# p1 from np.polyfit, plot(x, p1 with

# np.polyval(p1,x)= array([ 0.75714286, 0.45428571, 0.15142857, -0.15142857,-0.45428571, -0.75714286])

plt.plot(xp, np.polyval(p2,xp), 'b--') # --: dached line
# np.polyval(p2,x)=array([ 0.22142857,0.58, 0.27714286, -0.34714286, - 1.29285714])

plt.plot(xp, np.polyval(p3,xp), 'm:')
plt.plot(xp, np.polyval(p4,xp), 'g.')
plt.show()


#  new example

x = np.arange(10, 90, 10.)
y = np.array([25, 70, 380, 550, 610, 1220, 830, 1450])
plt.figure(5)
plt.plot(x, y, 'ro-')
plt.grid()
plt.show()
xsum=np.sum(x)
ysum=np.sum(y)
# 360.0
# 5135
xysum=sum(x*y)
n=np.size(x)
xavg=xsum/n # 45.0
yavg=ysum/n # 641.875
a1=(n*xysum-xsum*ysum)/(n*sum(x**2)-xsum**2)
# 19.470238095238095
a0= yavg-xavg*a1
# -234.28571428571422
y1=a1*x+a0
#array([-39.58333333, 155.11904762, 349.82142857, 544.52380952,
# 739.22619048, 933.92857143, 1128.63095238, 1323.33333333])
plt.figure(6)
plt.plot(x, y, 'ro-', x, y1, 'b*-')
plt.grid()
plt.show()
##########


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


plt.figure(7)
plt.plot(x,y, 'o')
plt.plot(x,fx, 'r*-')
plt.show()
plt.figure(8)
plt.plot(x,y, 'o')
plt.plot(x, np.polyval(p1, x), 'b*-')
# = plt.plot(x,fx, 'r*-')
plt.show()

plt.figure(9)
plt.plot(x,y, 'o')
plt.plot(x, np.polyval(p1,x), 'r*-')
plt.plot(x, np.polyval(p2,x), 'b>-')
plt.plot(x, np.polyval(p3,x), 'mx-')
plt.show()

plt.figure(10)
plt.plot(x, y, 'o')
plt.grid()
plt.show()
xp=np.linspace(-2, 6, 100)

plt.plot(xp, np.polyval(p1,xp), 'r-') # p1 from np.polyfit, plot(x, p1 with

# np.polyval(p1,x)= array([ 0.75714286, 0.45428571, 0.15142857, -0.15142857,-0.45428571,
# -0.75714286])
plt.plot(xp, np.polyval(p2,xp), 'b--') # --: dached line
# np.polyval(p2,x)=array([ 0.22142857,0.58, 0.27714286, -0.34714286,
# - 1.29285714])
plt.plot(xp, np.polyval(p3,xp), 'm:')
plt.show()


#  new example

x = np.arange(10, 90, 10.)
y = np.array([25, 70, 380, 550, 610, 1220, 830, 1450])
plt.figure(11)
plt.plot(x, y, 'ro:')
plt.grid()
plt.show()
xsum=np.sum(x)
ysum=np.sum(y)
# 360.0
# 5135
xysum=sum(x*y)
n=np.size(x)
xavg=xsum/n # 45.0
yavg=ysum/n # 641.875
a1=(n*xysum-xsum*ysum)/(n*sum(x**2)-xsum**2)
# 19.470238095238095
a0= yavg-xavg*a1
# -234.28571428571422
y1=a1*x+a0
#array([-39.58333333, 155.11904762, 349.82142857, 544.52380952,
# 739.22619048, 933.92857143, 1128.63095238, 1323.33333333])
plt.figure(12)
plt.plot(x, y, 'ro:', x, y1, 'b*-')
plt.grid()
plt.show()
#################################
x=np.array([0, 1, 2, 3, 4, 5])
y=np.array([0, 0.8, 0.9, 0.1, -0.8, -1])
xp=np.linspace(-2, 6, 100)


p1=np.polyfit(x, y, 1)
fx=p1[0]*x+p1[1]

plt.figure(13)
plt.plot(x, fx, 'r*-')
plt.show()
p2=np.polyfit(x, y, 2)
plt.plot(x, p2[0]*x**2+p2[1]*x+p2[2], 'b>:')

p3=np.polyfit(x, y, 3)
plt.plot(x, p3[0]*x**3+p3[1]*x**2+p3[2]*x+p3[3], 'mx:')


np.size(xp)

plt.figure(14)
plt.plot(x, y, 'o')
plt.plot(xp, p1[0]*xp+p1[1], 'r-')
plt.plot(xp, p2[0]*xp**2+p2[1]*xp+p2[2], 'b>:')
plt.plot(xp, p3[0]*xp**3+p3[1]*xp**2+p3[2]*xp+p3[3], 'm:')
plt.show()
plt.show()
x = np.arange(10, 90, 10.)
y = np.array([25, 70, 380, 550, 610, 1220, 830, 1450])

plt.figure(15)
plt.plot(x, y, 'ro-')
plt.grid()
plt.show()
xsum=np.sum(x)
ysum=np.sum(y)
# 360.0
# 5135
xysum=sum(x*y)
n=np.size(x)
xavg=xsum/n # 45.0
yavg=ysum/n # 641.875
a1=(n*xysum-xsum*ysum)/(n*sum(x**2)-xsum**2)
# 19.470238095238095
a0= yavg-xavg*a1
# -234.28571428571422
y1=a1*x+a0
#array([-39.58333333, 155.11904762, 349.82142857, 544.52380952,
#739.22619048, 933.92857143, 1128.63095238, 1323.33333333])
plt.figure(16)
plt.plot(x, y, 'ro-', x, y1, 'b*-')
plt.grid()
plt.show()



















