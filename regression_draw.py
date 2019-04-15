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


def Data_Learning(x_data, y_data, lo):
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b
    loss = tf.reduce_mean(tf.square(y - y_data))
    lo.append(loss)
    optimizer = tf.train.GradientDescentOptimizer(0.0015)
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

    return lo


if __name__ == '__main__':
    num_points=50
    x_data, y_data=Data_Genearion(num_points)
    Data_Draw(x_data, y_data)
    lo=[]
    ls=Data_Learning(x_data, y_data, lo)
    print(ls)
