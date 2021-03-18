

import numpy as np

# np.matrix() 이용한 행렬 내적 연산 
a=np.matrix([1,2,3])
b=np.matrix([4,5,6])
np.dot(a, b)

#ValueError: shapes (1,3) and (1,3) not aligned: 3 (dim 1) != 1 (dim 0)

b.transpose()
bt=b.transpose()
# p.matrix([[4],      [5],     [6]])

re=np.dot(a, bt)
#np.matrix([[32]])

# np.array() 이용한 행렬 내적 연산 
# 행렬 내적 연산 시 np.matrix() 보다는 np.array()를 많이 사용
# 왜냐면 np.matrix()를 사용하여 행렬 내적 연산을 하려면 뒤의 행렬을 transpose 전치
# 시켜야 하기 때문

a1=np.array([1,2,3])
b1=np.array([4,5,6])
re1=np.dot(a1, b1)
# 32

# 행렬 element by element 연산 
re2=a1*b1 
# Out[20]: array([ 4, 10, 18])

