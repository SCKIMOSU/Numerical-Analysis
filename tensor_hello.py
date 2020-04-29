#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 텐서플로우 2.0 환경에서 1.x 코드 실행하기 
# print(tf.__version__) 

def hello():
    a = tf.constant('hello, tensorflow!')
    print(a)  # Tensor("Const:0", shape=(), dtype=string)
    sess = tf.Session()
    result = sess.run(a)
    # 2.x 버전에서는 문자열로 출력되지만, 3.x 버전에서는 byte 자료형    # 문자열로 변환하기 위해 decode 함수로 변환
    print(result)  # b'hello, tensorflow!'
    print(type(result))  # <class 'bytes'>
    print(result.decode(encoding='utf-8'))  # hello, tensorflow!
    print(type(result.decode(encoding='utf-8')))  # <class 'str'>
    # 세션 닫기
    sess.close()

if '__name__' == '__main__':
    hello()

'''''''''
Tensor("Const_5:0", shape=(), dtype=string)
b'hello, tensorflow!'
<class 'bytes'>
hello, tensorflow!
<class 'str'>
'''
