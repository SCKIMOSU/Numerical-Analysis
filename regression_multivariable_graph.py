import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
# 텐서플로우 2.0 환경에서 1.x 코드 실행하기
# print(tf.__version__)
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop


raw_data = np.genfromtxt('x09.txt', skip_header=36)

xs = np.array(raw_data[:,2], dtype=np.float32) # weight
ys = np.array(raw_data[:,3], dtype=np.float32) # age
zs = np.array(raw_data[:,4], dtype=np.float32) # blood fat

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


###

x_data = np.array(raw_data[:,2:4], dtype=np.float32)
y_data = np.array(raw_data[:,4], dtype=np.float32)

y_data = y_data.reshape((25,1))

rmsprop=RMSprop(lr=0.01)

model=Sequential()
model.add(Dense(1, input_shape=(2,)))
model.compile(loss='mse', optimizer=rmsprop)
model.summary()

hist=model.fit(x_data, y_data, epochs=2000)

print(hist.history.keys())
# dict_keys(['loss'])

plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.predict(np.array([100, 40]).reshape(1,2))
model.predict(np.array([60, 25]).reshape(1,2))


W_, b_=model.get_weights()
W_, b_

x=np.linspace(20, 100, 50).reshape(50,1)
y=np.linspace(10, 70, 50).reshape(50,1)

X=np.concatenate((x,y), axis=1)
Z=np.matmul(X, W_)+b_

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, Z)
ax.scatter(xs, ys, zs)
ax.set_xlabel('Weight')
ax.set_ylabel('Age')
ax.set_zlabel('Blood fat')
ax.view_init(15, 15)

plt.show()


###########################
X = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cost)


# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# sess.run(W)

cost_history = []

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val)
        cost_history.append(sess.run(cost, feed_dict={X: x_data, Y: y_data}))
