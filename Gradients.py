# https://github.com/rickiepark/first-steps-with-tensorflow/blob/master/special_notes/Gradients.ipynb 

# https://icim.nims.re.kr/post/easyMath/70
# Learning rate가 0.5인 경우(발산) 
# 방정식 2*x = 10 을 만족하는 x 찾기
# x 초깃값 = 0

X = tf.Variable(0.)
Y = tf.constant(10.)
H = 2 * X


loss = tf.square(H-Y)
optimize = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sequence = []



for i in range(10):
    print("x_%i = %s" %(+i, sess.run(X)))
    sess.run(optimize)
    sequence.append(sess.run(X))

plt.suptitle("Sequence of x", fontsize=20)
plt.ylabel("x value")
plt.xlabel("Steps")
plt.plot(sequence, "o")



# https://pythonkim.tistory.com/9 
import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# try to find values for w and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# my hypothesis
hypothesis = W * x_data + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize
rate = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

# before starting, initialize the variables. We will 'run' this first.
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print('{:4} {} {} {}'.format(step, sess.run(cost), sess.run(W), sess.run(b)))

# learns best fit is W: [1] b: [0]



# 두 번째로 소개된 placeholder를 사용한 버전의 코드다.

출처: https://pythonkim.tistory.com/9 [파이쿵]
import tensorflow as tf

x_data = [1., 2., 3., 4.]
y_data = [2., 4., 6., 8.]

# range is -100 ~ 100
W = tf.Variable(tf.random_uniform([1], -100., 100.))
b = tf.Variable(tf.random_uniform([1], -100., 100.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

print(sess.run(hypothesis, feed_dict={X: 5}))           # [ 10.]
print(sess.run(hypothesis, feed_dict={X: 2.5}))         # [5.]
print(sess.run(hypothesis, feed_dict={X: [2.5, 5]}))    # [  5.  10.], 원하는 X의 값만큼 전달.





