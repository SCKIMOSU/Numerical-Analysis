# https://github.com/rickiepark/first-steps-with-tensorflow/blob/master/special_notes/Gradients.ipynb 
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

