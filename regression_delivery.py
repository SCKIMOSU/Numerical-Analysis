import numpy as np
from matplotlib import pyplot as plt

data = np.array([[100, 20],
		[150, 24],
		[300, 36],
		[400, 47],
		[130, 22],
		[240, 32],
		[350, 47],
		[200, 42],
		[100, 21],
		[110, 21],
		[190, 30],
		[120, 25],
		[130, 18],
		[270, 38],
		[255, 28]])

plt.scatter(data[:, 0], data[:, 1])
plt.title("Time / Distance")
plt.xlabel("Delivery Distance (meter)")
plt.ylabel("Time Consumed (minute)")
plt.axis([0, 420, 0, 50])
plt.show() 
