

import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 텐서플로우 2.0 환경에서 1.x 코드 실행하기
print(tf.__version__)


def Data_Genearion(num_points):
    #num_points = 50
    vectors_set = []
    for i in np.arange(num_points):
        x = np.random.normal(2, 2) + 10
        # [0,4] @ 68% + 10 =  [10,14] @ 68%
        y = x * 5 + (np.random.normal(0, 3)) * 2
        # [10,14] @ 68%  *5 = [50, 70] @ 68%
        # [50, 70] @ 68% + [-3, 3] @ 68% *2  = [44, 76] @ 68%
        vectors_set.append([x, y])

        # print(np.round(vectors_set, 1))

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]
    # print(np.round(x_data, 1))
    # print(np.round(y_data, 1))
    return  x_data, y_data



def Data_Draw(x_data, y_data):
    plt.plot(x_data, y_data,'ro')
    plt.ylim([0,100])
    plt.xlim([0,25])
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.legend()
    plt.show()


def Data_Learning(x_data, y_data):
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)
    # sess.run(W)
    # print('sess.run(W)= ', sess.run(W))
    # array([0.05211711], dtype=float32)

    b = tf.Variable(tf.zeros([1]))
    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)
    # sess.run(b)
    # array([0.], dtype=float32)

    y = W * x_data + b
    # sess.run(y)
    # print(np.round(sess.run(y),1))

    loss = tf.reduce_mean(tf.square(y - y_data))
    # print(np.round(sess.run(loss),1))
    # optimizer = tf.train.GradientDescentOptimizer(0.0015)
    #
    #optimizer = tf.train.GradientDescentOptimizer(0.0001)
    # 느린 수렴 : (0.0001) --> for step in np.arange(1000):
    #

    optimizer = tf.train.GradientDescentOptimizer(0.0015)
    # 적절한 수렴 : (0.0015) --> for step in np.arange(10):
    # 반복 연산복잡도를 줄이는 중요한 역할을 하는 학습률

    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    for step in np.arange(10):
        sess.run(train)
        print(step, sess.run(W), sess.run(b))
        print(step, sess.run(loss))
        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.legend()
        plt.show()


if __name__ == '__main__':
    num_points=50
    x_data, y_data=Data_Genearion(num_points)
    Data_Draw(x_data, y_data)
    Data_Learning(x_data, y_data)
