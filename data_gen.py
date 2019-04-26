import numpy as np
import matplotlib.pyplot as plt

num_points=50
vectors_set=[]

for i in np.arange(num_points):
    x1=np.random.normal(0.0, 0.55)
    y1=x1*0.1*0.3+np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data=[v[0] for v in vectors_set]
y_data=[v[1] for v in vectors_set]

plt.plot(x_data, y_data, 'ro')
plt.show()

p1=np.polyfit(x_data, y_data, 1)
plt.plot(x_data, np.polyval(p1, x_data), 'b*-')

p2=np.polyfit(x_data, y_data, 2)
plt.plot(x_data, np.polyval(p2, x_data), 'k>-')

p3=np.polyfit(x_data, y_data, 3)
plt.plot(x_data, np.polyval(p3, x_data), 'md-')
