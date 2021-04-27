import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 텐서플로우 2.0 환경에서 1.x 코드 실행하기
print(tf.__version__)

a=tf.constant(10)
b=tf.constant(32)
c=tf.add(a,b)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
