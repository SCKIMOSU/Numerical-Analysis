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
print('Est [100,40] : ', model.predict([[100,40]])) # (Weight:100, Age: 40) -> blood fat: 328.38
print('Est [60,25] : ', model.predict([[60,25]])) # (Weight:60, Age: 25) -> blood fat: 233.43


knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)

print(knn)
print('Est [100,40] : ', knn.predict([[100,40]])) # (Weight:100, Age: 40) -> blood fat: 306
print('Est [60,25] : ', knn.predict([[60,25]])) # (Weight:60, Age: 25) -> blood fat: 262


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


################################################
# 다음은 scikit-learn의 LinearRegression 클래스를 이용하여
# 보스턴 집값 데이터에 대해 회귀분석을 하는 예이다.

from sklearn.datasets import load_boston

boston = load_boston()
model_boston = sklearn.linear_model.LinearRegression().fit(boston.data, boston.target)

# 추정한 가중치 값은 다음과 같다. 특징 벡터의 이름과 비교하면
# 각각의 가중치가 가지는 의미를 알 수 있다.
# 예를 들어 방(RM) 하나가 증가하면 가격 예측치는
# 약 3,810달러 정도 증가한다는 것을 알 수 있다

model_boston.coef_
# array([-1.08011358e-01,  4.64204584e-02,  2.05586264e-02,  2.68673382e+00,
#        -1.77666112e+01,  3.80986521e+00,  6.92224640e-04, -1.47556685e+00,
#         3.06049479e-01, -1.23345939e-02, -9.52747232e-01,  9.31168327e-03,
#        -5.24758378e-01])


boston.feature_names
# array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
#        'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')

model_boston.intercept_
# 36.459488385089855

predictions = model_boston.predict(boston.data)

plt.figure(10)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# 그런데 보통 한글 글꼴에는 유니코드 마이너스(−)가 없고
# 일반 마이너스(-) 기호만 있습니다.
# 눈으로 보기에는 비슷해보이지만 다른 글자입니다.
# 따라서 유니코드 마이너스 기호를 쓰지 않도록 설정해줍니다.
plt.scatter(boston.target, predictions)
plt.xlabel(u"실제 집값")
plt.ylabel(u"집값 예측치")
plt.title("집값 예측치와 실제 집값의 관계")
plt.show()


