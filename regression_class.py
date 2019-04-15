

import tensorflow as tf

X = [1., 2., 3.]
Y = [1., 2., 3.]
m = len(X)

W = tf.placeholder(tf.float32)

#hypothesis = tf.mul(W, X)
hypothesis = W*X
cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2)) / m

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# 그래프로 표시하기 위해 데이터를 누적할 리스트
W_val, cost_val = [], []

# 0.1 단위로 증가할 수 없어서 -30부터 시작. 그래프에는 -3에서 5까지 표시됨.
for i in range(-30, 50):
    xPos = i*0.1                                    # x 좌표. -3에서 5까지 0.1씩 증가
    yPos = sess.run(cost, feed_dict={W: xPos})      # x 좌표에 따른 y 값

    print('{:3.1f}, {:3.1f}'.format(xPos, yPos))

    # 그래프에 표시할 데이터 누적. 단순히 리스트에 갯수를 늘려나감
    W_val.append(xPos)
    cost_val.append(yPos)

sess.close()

# ------------------------------------------ #

import matplotlib.pyplot as plt

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()


##-----------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def Data_Genearion(num_points):
    #num_points = 50
    vectors_set = []
    for i in np.arange(num_points):
        x = np.random.normal(2, 2) + 10
        y = x * 5 + (np.random.normal(0, 3)) * 2
        vectors_set.append([x, y])

        x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]

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
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b
    loss = tf.reduce_mean(tf.square(y - y_data))
    #lo.append(loss)
    optimizer = tf.train.GradientDescentOptimizer(0.0015)
    train = optimizer.minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    train_set = []  ###
    for step in np.arange(10):
        sess.run(train)
        print(step, sess.run(W), sess.run(b))
        print(step, sess.run(loss))
        train_set.append([sess.run(W), sess.run(b), sess.run(loss)])  ###

        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.legend()
        plt.show()

    W_data = [t[0] for t in train_set]
    v_data = [t[1] for t in train_set]
    Loss_data= [t[2] for t in train_set]

    return W_data,v_data, Loss_data


if __name__ == '__main__':
    num_points=50
    x_data, y_data=Data_Genearion(num_points)
    Data_Draw(x_data, y_data)

    W_data, v_data, Loss_data = Data_Learning(x_data, y_data)

    #for step in np.arange(10):

        #W_data[step]
    print('W_data = ', W_data)
    print('v_data = ', v_data)
    print('Loss_data = ', Loss_data)
