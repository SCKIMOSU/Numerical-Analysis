# 리스트 6-1-(1)
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
# 데이터 생성 --------------------------------
np.random.seed(seed=0) # 난수를 고정
X_min = 0
X_max = 2.5
X_n = 30
X_col = ['cornflowerblue', 'gray']
X = np.zeros(X_n) # 입력 데이터
T = np.zeros(X_n, dtype=np.uint8) # 목표 데이터
Dist_s = [0.4, 0.8] # 무게 분포의 시작 지점
Dist_w = [0.8, 1.6] # 무게 분포의 폭
Pi = 0.5 # 클래스 0의 비율
for n in range(X_n):
    wk = np.random.rand()
    T[n] = 0 * (wk < Pi) + 1 * (wk >= Pi) # (A)
    X[n] = np.random.rand() * Dist_w[T[n]] + Dist_s[T[n]] # (B)
# 데이터 표시 --------------------------------
print('X=' + str(np.round(X, 2))) # 곤충의 무게 출력
print('T=' + str(T)) # 곤충의 성별 출력
# X=[1.67 0.92 1.11 1.41 1.65 2.28 0.47 1.07 2.19 2.08 1.02 0.91 1.16 1.46
#  1.02 0.85 0.89 1.79 1.89 0.75 0.9  1.87 0.5  0.69 1.5  0.96 0.53 1.21
#  0.6  0.49]
# T=[1 0 0 1 1 1 0 0 1 1 0 0 0 1 0 0 0 1 1 0 1 1 0 0 1 1 0 1 0 0]

# 리스트 6-1-(2)
# 무게, 성별 데이터 분포 표시 ----------------------------
def show_data1(x, t): # x=X; t=T
    K = np.max(t) + 1
    for k in range(K): # (A)
        plt.plot(x[t == k], t[t == k], X_col[k], alpha=0.5,
                 linestyle='none', marker='o') # (B)
        plt.grid(True)
        plt.ylim(-.5, 1.5)
        plt.xlim(X_min, X_max)
        plt.yticks([0, 1])

# female=np.round(x[t == k], 2)
# array([0.92, 1.11, 0.47, 1.07, 1.02, 0.91, 1.16, 1.02, 0.85, 0.89, 0.75,
#       0.5 , 0.69, 0.53, 0.6 , 0.49])
# np.round(min(x[t == k]), 2) = 0.47
#fe_min=np.round(min(x[t == k]), 2)
#np.where(female==fe_min)
# (array([2], dtype=int64),)
# fe_max=np.round(max(x[t == k]), 2)
# np.where(female==fe_max)
#  (array([6], dtype=int64),)

# male=np.round(x[t == k], 2)
# array([1.67, 1.41, 1.65, 2.28, 2.19, 2.08, 1.46, 1.79, 1.89, 0.9 , 1.87,
#        1.5 , 0.96, 1.21])
# ma_min=np.round(min(x[t == k]), 2)
# 0.9
# ma_max=np.round(max(x[t == k]), 2)
# 2.28
# np.where(male==ma_min)
# (array([9], dtype=int64),)
# np.where(male==ma_max)
# (array([3], dtype=int64),)

# 메인 ------------------------------------
fig = plt.figure(figsize=(5, 5))
show_data1(X, T)
plt.show()


# 리스트 6-1-(3)
def logistic(x, w): # x=xb, w=W
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y

# 리스트 6-1-(4)
def show_logistic(w): # w=W
    xb = np.linspace(X_min, X_max, 100) # X_min=0, X_max=2.5
    # np.round(xb,2)
    # np.round(xb,2)
    # array([0.  , 0.03, 0.05, 0.08, 0.1 , 0.13, 0.15, 0.18, 0.2 , 0.23, 0.25,
    #        0.28, 0.3 , 0.33, 0.35, 0.38, 0.4 , 0.43, 0.45, 0.48, 0.51, 0.53,
    #        0.56, 0.58, 0.61, 0.63, 0.66, 0.68, 0.71, 0.73, 0.76, 0.78, 0.81,
    #        0.83, 0.86, 0.88, 0.91, 0.93, 0.96, 0.98, 1.01, 1.04, 1.06, 1.09,
    #        1.11, 1.14, 1.16, 1.19, 1.21, 1.24, 1.26, 1.29, 1.31, 1.34, 1.36,
    #        1.39, 1.41, 1.44, 1.46, 1.49, 1.52, 1.54, 1.57, 1.59, 1.62, 1.64,
    #        1.67, 1.69, 1.72, 1.74, 1.77, 1.79, 1.82, 1.84, 1.87, 1.89, 1.92,
    #        1.94, 1.97, 1.99, 2.02, 2.05, 2.07, 2.1 , 2.12, 2.15, 2.17, 2.2 ,
    #        2.22, 2.25, 2.27, 2.3 , 2.32, 2.35, 2.37, 2.4 , 2.42, 2.45, 2.47,
    #        2.5 ])
    y = logistic(xb, w)
    # np.round(y,2)
    # np.round(y,2)
    # Out[1126]:
    # array([0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
    #        0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
    #        0.  , 0.  , 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03,
    #        0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.21,
    #        0.25, 0.29, 0.33, 0.38, 0.42, 0.47, 0.53, 0.58, 0.62, 0.67, 0.71,
    #        0.75, 0.79, 0.82, 0.85, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95, 0.96,
    #        0.97, 0.97, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 1.  ,
    #        1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
    #        1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
    #        1.  ])
    plt.plot(xb, y, color='gray', linewidth=4)
    # 결정 경계
    i = np.min(np.where(y > 0.5)) # (A)
    # np.where(np.round(y,2) > 0.5)
    # (array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
    #         67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    #         84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
    #        dtype=int64),)

    # np.min(np.where(np.round(y,2) > 0.5)) : 50

    B = (xb[i - 1] + xb[i]) / 2 # (B)
    # xb[i - 1] :  1.23
    # xb[i] :  1.2626
    # B : 1.25

    plt.plot([B, B], [-.5, 1.5], color='k', linestyle='--')
    plt.grid(True)
    return B


# test
W = [8, -10]
show_logistic(W)


# 리스트 6-1-(5)
# 평균 교차 엔트로피 오차 ---------------------
def cee_logistic(w, x, t): # w=W; x=X; t=T # x: 질량, t: 암컷, 수컷
    y = logistic(x, w) # 로지스틱 함수, 확률 함수 (0~1)로 변환
    # logistic(x, w):
    #     y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n] * np.log(y[n]) + (1 - t[n]) * np.log(1 - y[n]))
        # t
        # array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1,
        #        1, 0, 0, 1, 1, 0, 1, 0], dtype=uint8)
        # np.round(y,2) : 로지스틱 확률이 최대가 되는 w0와 w1값을 계산한다. 로그가능도
        #
        # array([1.  , 0.97, 0.06, 0.25, 0.79, 0.96, 1.  , 0.  , 0.19, 1.  , 1.  ,
        #        0.14, 0.06, 0.32, 0.85, 0.14, 0.04, 0.05, 0.99, 0.99, 0.02, 0.06,
        #        0.99, 0.  , 0.01, 0.88, 0.09, 0.  , 0.41, 0.01])
        # 로그가능도 (최대를 찾는)가 cee로 (교차엔트로피함수, 최소를 찾는)계산으로 바뀜으로써,
        # 교차엔트로피함수가 최소가 되도록 하는 w0와 w1값을 계산한다
    cee = cee / X_n
    print(cee)
    return cee


# test
W=[1,1]
cee_logistic(W, X, T)


# 리스트 6-1-(6)
from mpl_toolkits.mplot3d import Axes3D


# 계산 --------------------------------------
xn = 80 # 등고선 표시 해상도
w_range = np.array([[0, 15], [-15, 0]])
x0 = np.linspace(w_range[0, 0], w_range[0, 1], xn)
x1 = np.linspace(w_range[1, 0], w_range[1, 1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
C = np.zeros((len(x1), len(x0)))
w = np.zeros(2)
for i0 in range(xn):
    for i1 in range(xn):
        w[0] = x0[i0]
        w[1] = x1[i1]
        C[i1, i0] = cee_logistic(w, X, T)


# 표시 --------------------------------------
plt.figure(figsize=(12, 5))
#plt.figure(figsize=(9.5, 4))
plt.subplots_adjust(wspace=0.5)
ax = plt.subplot(1, 2, 1, projection='3d')
ax.plot_surface(xx0, xx1, C, color='blue', edgecolor='black',
                rstride=10, cstride=10, alpha=0.3)
ax.set_xlabel('$w_0$', fontsize=14)
ax.set_ylabel('$w_1$', fontsize=14)
ax.set_xlim(0, 15)
ax.set_ylim(-15, 0)
ax.set_zlim(0, 8)
ax.view_init(30, -95)


plt.subplot(1, 2, 2)
cont = plt.contour(xx0, xx1, C, 20, colors='black',
                   levels=[0.26, 0.4, 0.8, 1.6, 3.2, 6.4])
cont.clabel(fmt='%1.1f', fontsize=8)
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)
plt.grid(True)
plt.show()


# 리스트 6-1-(7)
# 평균 교차 엔트로피 오차의 미분 --------------
def dcee_logistic(w, x, t):
    y = logistic(x, w)
    dcee = np.zeros(2)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n]) * x[n]
        dcee[1] = dcee[1] + (y[n] - t[n])
    dcee = dcee / X_n
    print('dcee= ', dcee) # by sck 2020_02_11
    return dcee


# --- test
W=[1, 1]
dcee_logistic(W, X, T)


# 리스트 6-1-(8)
from scipy.optimize import minimize


# 매개 변수 검색
def fit_logistic(w_init, x, t):
    res1 = minimize(cee_logistic, w_init, args=(x, t),
                    jac=dcee_logistic, method="CG") # (A)
    return res1.x


# 메인 ------------------------------------
plt.figure(1, figsize=(3, 3))
W_init=[1,-1]
W = fit_logistic(W_init, X, T)
print("w0 = {0:.2f}, w1 = {1:.2f}".format(W[0], W[1]))
B=show_logistic(W)
show_data1(X, T)
plt.ylim(-.5, 1.5)
plt.xlim(X_min, X_max)
cee = cee_logistic(W, X, T)
print("CEE = {0:.2f}".format(cee))
print("Boundary = {0:.2f} g".format(B))
plt.show()

# reset

# 리스트 6-2-(1)
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline


# 데이터 생성 --------------------------------
np.random.seed(seed=1)  # 난수를 고정
N = 100 # 데이터의 수
K = 3 # 분포 수
T3 = np.zeros((N, 3), dtype=np.uint8)
T2 = np.zeros((N, 2), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3] # X0 범위 표시 용
X_range1 = [-3, 3] # X1의 범위 표시 용
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]]) # 분포의 중심
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]]) # 분포의 분산
Pi = np.array([0.4, 0.8, 1]) # (A) 각 분포에 대한 비율 0.4 0.8 1
for n in range(N):
    wk = np.random.rand()
    for k in range(K): # (B)
        if wk < Pi[k]:
            T3[n, k] = 1
            break
    for k in range(2):
        X[n, k] = (np.random.randn() * Sig[T3[n, :] == 1, k]
                   + Mu[T3[n, :] == 1, k])
T2[:, 0] = T3[:, 0]
T2[:, 1] = T3[:, 1] | T3[:, 2]

# 리스트 6-2-(2)
print(X[:5,:])

# 리스트 6-2-(3)
print(T2[:5,:])

# 리스트 6-2-(4)
print(T3[:5,:])

# 리스트 6-2-(5)
# 데이터 표시 --------------------------
def show_data2(x, t):
    wk, K = t.shape
    c = [[.5, .5, .5], [1, 1, 1], [0, 0, 0]]
    for k in range(K):
        temp1=x[t[:, k] == 1, 0] # sckim 2020 02 21
        temp2=x[t[:, k] == 1, 1] # sckim 2020 02 21
        temp=np.array([temp1, temp2]) # sckim
        plt.plot(x[t[:, k] == 1, 0], x[t[:, k] == 1, 1],
                 linestyle='none', markeredgecolor='black',
                 marker='o', color=c[k], alpha=0.8)
        #plt.show() # sck 2020 2 14
        plt.grid(True)


# 메인 ------------------------------

plt.figure(figsize=(7.5, 3))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1, 2, 1)
show_data2(X, T2)
plt.xlim(X_range0)
plt.ylim(X_range1)


plt.subplot(1, 2, 2)
show_data2(X, T3)
plt.xlim(X_range0)
plt.ylim(X_range1)



# 리스트 6-2-(6)
# 로지스틱 회귀 모델 -----------------
def logistic2(x0, x1, w):
    y = 1 / (1 + np.exp(-(w[0] * x0 + w[1] * x1 + w[2])))
    return y


# 리스트 6-2-(7)
# 모델 3D보기 ------------------------------
from mpl_toolkits.mplot3d import axes3d


def show3d_logistic2(ax, w):
    xn = 50
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    #plt.show()  # sckim 2020 2 21
    y = logistic2(xx0, xx1, w)
    ax.plot_surface(xx0, xx1, y, color='blue', edgecolor='gray',
                    rstride=5, cstride=5, alpha=0.3)
    #plt.show()  # sckim 2020 2 21


def show_data2_3d(ax, x, t):
    c = [[.5, .5, .5], [1, 1, 1]]
    for i in range(2):
        ax.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], 1 - i,
                marker='o', color=c[i], markeredgecolor='black',
                linestyle='none', markersize=5, alpha=0.8)
        #plt.show()  # sckim 2020 2 21
    Ax.view_init(elev=25, azim=-30)
    #plt.show() # sckim 2020 2 21


# test ---
Ax = plt.subplot(1, 1, 1, projection='3d')
W=[-1, -1, -1]
show3d_logistic2(Ax, W)
show_data2_3d(Ax,X,T2)

plt.show()


# 리스트 6-2-(8)
# 모델 등고선 2D 표시 ------------------------


def show_contour_logistic2(w):
    xn = 30 # 파라미터의 분할 수
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    #plt.show()  # sckim 2020 2 21
    y = logistic2(xx0, xx1, w)
    cont = plt.contour(xx0, xx1, y, levels=(0.2, 0.5, 0.8),
                       colors=['k', 'cornflowerblue', 'k'])
    #plt.show()  # sckim 2020 2 21
    cont.clabel(fmt='%1.1f', fontsize=10)
    #plt.show()  # sckim 2020 2 21
    plt.grid(True)


# test ---
plt.figure(figsize=(3,3))
W=[-1, -1, -1]
show_contour_logistic2(W)

# 리스트 6-2-(9)
# 크로스 엔트로피 오차 ------------
def cee_logistic2(w, x, t):
    X_n = x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n, 0] * np.log(y[n]) +
                     (1 - t[n, 0]) * np.log(1 - y[n]))
        #print(cee) # sckim 2020 2 22
    cee = cee / X_n
    return cee

# 리스트 6-2-(10)
# 크로스 엔트로피 오차의 미분 ------------
def dcee_logistic2(w, x, t):
    X_n=x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    dcee = np.zeros(3)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n, 0]) * x[n, 0]
        dcee[1] = dcee[1] + (y[n] - t[n, 0]) * x[n, 1]
        dcee[2] = dcee[2] + (y[n] - t[n, 0])
    dcee = dcee / X_n
    return dcee


# test ---
W=[-1, -1, -1]
print(dcee_logistic2(W, X, T2)) # sckim 2020 02 25


# 리스트 6-2-(11)
from scipy.optimize import minimize


# 로지스틱 회귀 모델의 매개 변수 검색 -
def fit_logistic2(w_init, x, t):
    res = minimize(cee_logistic2, w_init, args=(x, t),
                   jac=dcee_logistic2, method="CG")
    return res.x


# 메인 ------------------------------------
plt.figure(1, figsize=(7, 3))
plt.subplots_adjust(wspace=0.5)


Ax = plt.subplot(1, 2, 1, projection='3d')
W_init = [-1, 0, 0]
W = fit_logistic2(W_init, X, T2)
print("w0 = {0:.2f}, w1 = {1:.2f}, w2 = {2:.2f}".format(W[0], W[1], W[2]))
show3d_logistic2(Ax, W)


show_data2_3d(Ax, X, T2)
cee = cee_logistic2(W, X, T2)
print("CEE = {0:.2f}".format(cee))


Ax = plt.subplot(1, 2, 2)
show_data2(X, T2)
show_contour_logistic2(W)
plt.show()


# 리스트 6-2-(12)
# 3 클래스 용 로지스틱 회귀 모델 -----------------


def logistic3(x0, x1, w):
    K = 3
    w = w.reshape((3, 3))
    n = len(x1)
    y = np.zeros((n, K))
    for k in range(K):
        y[:, k] = np.exp(w[k, 0] * x0 + w[k, 1] * x1 + w[k, 2])
    wk = np.sum(y, axis=1)
    wk = y.T/ wk
    y = wk.T
    return y


# test ---
W = np.array([1, 2, 3, 4 ,5, 6, 7, 8, 9])
y = logistic3(X[:3, 0], X[:3, 1], W)
print(np.round(y, 3))


# 리스트 6-2-(13)
# 교차 엔트로피 오차 ------------
def cee_logistic3(w, x, t):
    X_n = x.shape[0]
    y = logistic3(x[:, 0], x[:, 1], w)
    cee = 0
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            cee = cee - (t[n, k] * np.log(y[n, k]))
    cee = cee / X_n
    return cee


# test ----
W = np.array([1, 2, 3, 4 ,5, 6, 7, 8, 9])
print(cee_logistic3(W, X, T3)) # sckim 2020 2 29

# 리스트 6-2-(14)
# 교차 엔트로피 오차의 미분 ------------
def dcee_logistic3(w, x, t):
    X_n = x.shape[0]
    y = logistic3(x[:, 0], x[:, 1], w)
    dcee = np.zeros((3, 3)) # (클래스의 수 K) x (x의 차원 D+1)
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            dcee[k, :] = dcee[k, :] - (t[n, k] - y[n, k])* np.r_[x[n, :], 1]
    dcee = dcee / X_n
    return dcee.reshape(-1)


# test ----
W = np.array([1, 2, 3, 4 ,5, 6, 7, 8, 9]) # random W
print(dcee_logistic3(W, X, T3)) # sckim 2020 3 1  # random W  --> minimize W


# 리스트 6-2-(15)
# 매개 변수 검색 -----------------
def fit_logistic3(w_init, x, t):
    res = minimize(cee_logistic3, w_init, args=(x, t),
                   jac=dcee_logistic3, method="CG")
    return res.x


# 리스트 6-2-(16)
# 모델 등고선 2D 표시 --------------------
def show_contour_logistic3(w):
    xn = 30 # 파라미터의 분할 수
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)


    xx0, xx1 = np.meshgrid(x0, x1)
    y = np.zeros((xn, xn, 3))
    for i in range(xn):
        wk = logistic3(xx0[:, i], xx1[:, i], w)
        for j in range(3):
            y[:, i, j] = wk[:, j]
    for j in range(3):
        cont = plt.contour(xx0, xx1, y[:, :, j],
                           levels=(0.5, 0.9),
                           colors=['cornflowerblue', 'k'])
        cont.clabel(fmt='%1.1f', fontsize=9)
    plt.grid(True)



# 리스트 6-2-(17)
# 메인 ------------------------------------
W_init = np.zeros((3, 3))
W = fit_logistic3(W_init, X, T3)
print(np.round(W.reshape((3, 3)),2))
cee = cee_logistic3(W, X, T3)
print("CEE = {0:.2f}".format(cee))


plt.figure(figsize=(3, 3))
show_data2(X, T3)
show_contour_logistic3(W)
#show_data2_3d(ax, X, T3) # SCKIM 2020 3 2
plt.show()

'''''''''
def show_data2_3class_3d(ax, x, t): #  sckim  2020 3 2
   c = [[.5, .5, .5], [1, 1, 1]]
   for i in range(3): # range(2) --> range(3) sckim  2020 3 2
       ax.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], 1 - i,
               marker='o', color=c[i], markeredgecolor='black',
               linestyle='none', markersize=5, alpha=0.8)
       plt.show()  # sckim 2020 2 21
   Ax.view_init(elev=25, azim=-30)
   plt.show() # sckim 2020 2 21


plt.figure(100)
Ax = plt.subplot(1, 1, 1, projection='3d')
#show3d_logistic2(Ax, W)
show_data2_3class_3d(Ax,X,T3) # sckim 2020 2 21
plt.show()

'''
