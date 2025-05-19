import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 또는 'QtAgg', 'Agg'도 가능


# np.polyfit(x,y,2)를 이용한 다항 회귀
x=np.array([0,1,2,3,4,5])
y=np.array([2.1, 7.7, 13.6, 27.2, 40.9, 61.1])

p2=np.polyfit(x,y,2)
# array([1.86071429, 2.35928571, 2.47857143])
plt.figure(1)
plt.plot(x, y, 'ro', np.polyval(p2, x), 'b*-' )  # x, p2[0]*x**2+p2[1]*x+p2[2]
plt.legend(['Real Data', 'Polynomial Regression by np.polyfit(x,y,2)'])
plt.grid()
plt.show()


# 편미분을 이용한 다항 회귀
a=np.array([[np.size(x), np.sum(x), np.sum(x**2)], [np.sum(x), np.sum(x**2), np.sum(x**3) ], [np.sum(x**2), np.sum(x**3), np.sum(x**4) ] ])
#array([[  6,  15,  55],
#       [ 15,  55, 225],
#       [ 55, 225, 979]])
b=np.array([np.sum(y), np.sum(x*y), np.sum(x**2*y)])
# array([ 152.6,  585.6, 2488.8])
t=np.linalg.solve(a, b)
# array([2.47857143, 2.35928571, 1.86071429])

x1=np.linspace(0,5,10)
y1=t[2]*x1**2+t[1]*x1+t[0]
# derived from ploynomial regression
plt.figure(2)
plt.plot(x,y,'ro', x1, y1, 'b*:')
plt.legend(['Real Data','Polynomial Regression by Partial Derivative'])
plt.grid()
plt.show()
