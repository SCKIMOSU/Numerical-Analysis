# GradientDescentOptimizer
# 파이썬(3.7.3)언어로 작성한 경사 하강법 알고리즘은 f(x)=x4−3x3+2 함수의 극값을
# 미분값인 f'(x)=4x3−9x2 를 통해 찾는 예를 보여준다.[2]

# # From calculation, we expect that the local minimum occurs at x=9/4
#
x_old = 0
x_new = 6 # The algorithm starts at x=6
eps = 0.01 # step size
precision = 0.00001

def f_prime(x):
    return 4 * x**3 - 9 * x**2

while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new = x_old - eps * f_prime(x_old)

print("Local minimum occurs at: " + str(x_new))
