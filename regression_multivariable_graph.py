import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.neighbors import KNeighborsRegressor

from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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


plt.figure(1)
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
# 아래의 텐서플로우 코드를 두 개의 라인으로 추상화했음
# 추상화가 잘 된 케이스

# hypothesis = W1 * x1_data + W2 * x2_data + b
# cost = tf.reduce_mean(tf.square(hypothesis - y_data))
# a = tf.Variable(0.1) #alpha, learning rate
# optimizer = tf.train.GradientDescentOptimizer(a)
# train = optimizer.minimize(cost)
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print (step, sess.run(cost), sess.run(W1),
#                sess.run(W2), sess.run(b) )
print(model)
print('Est [100,40] : ', model.predict([[100,40]]))
# 예측가능

# (Weight:100, Age: 40) -> blood fat: 328.38
# [328.38238085] -> sklearn LinearRegression 예측값
print('Est [60,25] : ', model.predict([[60,25]]))
#  (Weight:60, Age: 25) -> blood fat: 233.43
#   [233.43903476] -> sklearn LinearRegression 예측값


knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
# model = sklearn.linear_model.LinearRegression() 모델링과 다른
# KNeighborsRegressor(n_neighbors=3) KNeighborsRegressor 사용
knn.fit(X, y)

print(knn)
print('Est [100,40] : ', knn.predict([[100,40]]))
# (Weight:100, Age: 40) -> blood fat: 306
# [306.]  ->  sklearn KNeighborsRegressor 예측값
print('Est [60,25] : ', knn.predict([[60,25]]))
# (Weight:60, Age: 25) -> blood fat: 262
#  [262.] -> sklearn KNeighborsRegressor 예측값

###

x_data = np.array(raw_data[:,2:4], dtype=np.float32)
y_data = np.array(raw_data[:,4], dtype=np.float32)

y_data = y_data.reshape((25,1))

#from keras.models import Sequential
#from keras.layers.core import Dense
#from keras.optimizers import RMSprop

rmsprop=RMSprop(lr=0.01)
#
# https://keras.io/ko/getting-started/sequential-model-guide/
# Sequential 모델은 레이어를 선형으로 연결하여 구성합니다.
# 레이어 인스턴스를 생성자에게 넘겨줌으로써 Sequential 모델을 구성할 수 있습니다.

model=Sequential()

# 또한, .add() 메소드를 통해서 쉽게 레이어를 추가할 수 있습니다.
model.add(Dense(1, input_shape=(2,)))
# 모델을 학습시키기 이전에, compile 메소드를 통해서 학습 방식에 대한 환경설정을 해야 합니다.
# 다음의 세 개의 인자를 입력으로 받습니다.
model.compile(loss='mse', optimizer=rmsprop)
model.summary()

# 학습
# 케라스 모델들은 입력 데이터와 라벨로 구성된 Numpy 배열 위에서 이루어집니다. \
# 모델을 학습기키기 위해서는 일반적으로 fit함수를 사용합니다.
# 여기서 자세한 정보를 알 수 있습니다.

hist=model.fit(x_data, y_data, epochs=2000)

print(hist.history.keys())
# dict_keys(['loss'])
plt.figure(3)
plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()

model.predict(np.array([100, 40]).reshape(1,2))
# array([[348.74396]], dtype=float32) -> 케라스 예측값

# [328.38238085] -> sklearn LinearRegression 예측값
#  [306.]  ->  sklearn KNeighborsRegressor 예측값

model.predict(np.array([60, 25]).reshape(1,2))
# array([[222.32545]], dtype=float32) -> 케라스 예측값

# [233.43903476] -> sklearn LinearRegression 예측값
# [262.] -> sklearn KNeighborsRegressor 예측값

W_, b_=model.get_weights()
W_, b_
# 모델링된 가중치(기울기)와 절편을 출력해보자
# (array([[1.0955558], : 가중치
#         [5.507313 ]], : 절편 dtype=float32), array([18.904305], dtype=float32))

# 실제 데이터셋에 모델링된 직선을 그려보자
# 실제 데이터셋
x=np.linspace(20, 100, 50).reshape(50,1)
y=np.linspace(10, 70, 50).reshape(50,1)

X=np.concatenate((x,y), axis=1)

# 모델링된 모델값
Z=np.matmul(X, W_)+b_

plt.figure(4)

fig = plt.figure(figsize=(12,12))
# 3차원 구조를 만들어 보자
ax = fig.add_subplot(111, projection='3d')

# 모델링직선을 출력해보자
ax.scatter(x, y, Z)

# 실제 데이터셋을 출력해보자
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




###########


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

plt.figure(20)
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






