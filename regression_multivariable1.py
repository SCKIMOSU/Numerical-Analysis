
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
# 텐서플로우 2.0 환경에서 1.x 코드 실행하기
# print(tf.__version__)



#data
x1_data = [1,0,3,0,5]
x2_data = [0,2,0,4,0]
y_data = [1,2,3,4,5]

# W1=[1.0000001] W2=[1.0000001] b=[-2.9742586e-07]
W1 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
W2 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

#hypothesis
hypothesis = W1 * x1_data + W2 * x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#minimize
a = tf.Variable(0.1) #alpha, learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# before starting, initialize variables
init = tf.initialize_all_variables()
#launch
sess = tf.Session()
sess.run(init)
# fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(cost), sess.run(W1),
               sess.run(W2), sess.run(b) )
