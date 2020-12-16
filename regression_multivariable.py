import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
#import sklearn.model_selection
# from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsRegressor

from mpl_toolkits.mplot3d import Axes3D

raw_data = np.genfromtxt('x09.txt', skip_header=36)

xs = np.array(raw_data[:,2], dtype=np.float32)
ys = np.array(raw_data[:,3], dtype=np.float32)
zs = np.array(raw_data[:,4], dtype=np.float32)

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)
ax.set_xlabel('Weight')
ax.set_ylabel('Age')
ax.set_zlabel('Blood fat')
ax.view_init(15, 15)

plt.show()

X = np.array(raw_data[:,2:4], dtype=np.float64)
# 두 개의 변수가 사용된다.
# ax.set_xlabel('Weight')
# ax.set_ylabel('Age')
y = np.array(raw_data[:,4], dtype=np.float64)

model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

print(model)
print('Est [100,40] : ', model.predict([[100,40]]))
print('Est [60,25] : ', model.predict([[60,25]]))



knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)

print(knn)
print('Est [100,40] : ', knn.predict([[100,40]]))
print('Est [60,25] : ', knn.predict([[60,25]]))
