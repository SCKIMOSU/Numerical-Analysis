#-- 리스트 7-1-(1)
import numpy as np
# 데이터 생성 --------------------------------
np.random.seed(seed=1) # 난수를 고정
N = 200 # 데이터의 수
K = 3 # 분포의 수
T = np.zeros((N, 3), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3] # X0의 범위, 표시용
X_range1 = [-3, 3] # X1의 범위, 표시용
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]]) # 분포의 중심
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]]) # 분포의 분산
Pi = np.array([0.4, 0.8, 1]) # 각 분포에 대한 비율
for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            T[n, k] = 1
            break
    for k in range(2):
        X[n, k] = np.random.randn() * Sig[T[n, :] == 1, k] + \
        Mu[T[n, :] == 1, k]


print('X= :', X)   #sckim 2020 3 2  # 입력은 2개 이지만,
print('T= :', T)   #sckim 2020 3 2  # 클래스는 3개로 분류됨


#-- 리스트 7-1-(2)
# -------- 2 분류 데이터를 테스트 훈련 데이터로 분할
TestRatio = 0.5
X_n_training = int(N * TestRatio)
X_train = X[:X_n_training, :]
X_test = X[X_n_training:, :]
T_train = T[:X_n_training, :]
T_test = T[X_n_training:, :]


# -------- 데이터를 'class_data.npz'에 저장
np.savez('class_data.npz', X_train=X_train, T_train=T_train,
         X_test=X_test, T_test=T_test,
         X_range0=X_range0, X_range1=X_range1)


#-- 리스트 7-1-(3)
import matplotlib.pyplot as plt
#matplotlib inline


# 데이터를 그리기 ------------------------------
def Show_data(x, t):
    wk, n = t.shape
    c = [[0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    for i in range(n):
        plt.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1],
                 linestyle='none',
                 marker='o', markeredgecolor='black',
                 color=c[i], alpha=0.8)
    plt.grid(True)


# 메인 ------------------------------------
plt.figure(1, figsize=(8, 3.7))
plt.subplot(1, 2, 1)
Show_data(X_train, T_train)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Training Data')
plt.subplot(1, 2, 2)
Show_data(X_test, T_test)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.title('Test Data')
plt.show()


# 리스트 7-1-(4)
# 시그모이드 함수 ------------------------
def Sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


# 네트워크 ------------------------
def FNN(wv, M, K, x): # wv=WV; M; K; x=X_train[:2, :]
    # M=2; K=3; D=2
    # np.round(X_train[:2, :], 2)
    # array([[-0.14,  0.87],
    #        [-0.87, -1.25]])

    N, D = x.shape # 입력 차원
    w = wv[:M * (D + 1)] # 중간층 뉴런의 가중치
    # M * (D + 1) = 2*3=6
    # w = array([1., 1., 1., 1., 1., 1.])
    w = w.reshape(M, (D + 1))
    # w
    # array([[1., 1., 1.],
    #        [1., 1., 1.]])

    v = wv[M * (D + 1):] # 출력층 뉴런의 가중치
    # v , M = 2 , D = 2
    # array([1., 1., 1., 1., 1., 1., 1., 1., 1.])

    v = v.reshape((K, M + 1))
    # v, K=3, M=2
    # array([[1., 1., 1.],
    #        [1., 1., 1.],
    #        [1., 1., 1.]])

    b = np.zeros((N, M + 1)) # 중간층 뉴런의 입력 총합
    # b , N=2, M=2
    # array([[0., 0., 0.],
    #        [0., 0., 0.]])

    z = np.zeros((N, M + 1)) # 중간층 뉴런의 출력
    # z , N=2, M=2
    # array([[0., 0., 0.],
    #        [0., 0., 0.]])

    a = np.zeros((N, K)) # 출력층 뉴런의 입력 총합
    # a, N=2, K=3
    # array([[0., 0., 0.],
    #        [0., 0., 0.]])

    y = np.zeros((N, K)) # 출력층 뉴런의 출력
    # y, N=2, K=3
    # array([[0., 0., 0.],
    #        [0., 0., 0.]])

    for n in range(N):  # N=2
        # 중간층의 계산
        for m in range(M): # M=2
            b[n, m] = np.dot(w[m, :], np.r_[x[n, :], 1]) # (A)
            # np.r_[x[n, :], 1]
            # array([-0.14,  0.87,  1.])

            # b
            # array([[0., 0., 0.],
            #        [0., 0., 0.]])
            # w[m, :] = array([1., 1., 1.])
            # x[n, :] = array([-0.14,  0.87])

            z[n, m] = Sigmoid(b[n, m])
            # z
            # array([[0.5, 0. , 0. ],
            #        [0. , 0. , 0. ]])

        # 출력층의 계산
        z[n, M] = 1 # 더미 뉴런
        wkz = 0
        for k in range(K):
            a[n, k] = np.dot(v[k, :], z[n, :])
            wkz = wkz + np.exp(a[n, k])
        for k in range(K):
            y[n, k] = np.exp(a[n, k]) / wkz
        #y  =np.round(y, 2)
        #a = np.round(a, 2)
        #z = np.round(z, 2)
        #b = np.round(b, 2)
    return y, a, z, b
# np.round(y, 2)
# array([[0.33, 0.33, 0.33],
#        [0.33, 0.33, 0.33]])
# np.round(a, 2)
# array([[2.7 , 2.7 , 2.7 ],
#        [1.49, 1.49, 1.49]])
# np.round(z, 2)
# array([[0.85, 0.85, 1.  ],
#        [0.25, 0.25, 1.  ]])
# np.round(b, 2)
# array([[ 1.72,  1.72,  0.  ],
#        [-1.12, -1.12,  0.  ]])

# test ---
WV = np.ones(15)
M = 2
K = 3
y, a, z, b=FNN(WV, M, K, X_train[:2, :])
# FNN(WV, M, K, X_train[:2, :])
# y
# array([[0.33, 0.33, 0.33],
#        [0.33, 0.33, 0.33]])
# a
# array([[2.7 , 2.7 , 2.7 ],
#        [1.49, 1.49, 1.49]])
# z
# array([[0.85, 0.85, 1.  ],
#        [0.25, 0.25, 1.  ]])
# b
# array([[ 1.72,  1.72,  0.  ],
#        [-1.12, -1.12,  0.  ]])

# 리스트 7-1-(5)
# 평균 교차 엔트로피 오차 ---------
def CE_FNN(wv, M, K, x, t):
    N, D = x.shape
    y, a, z, b = FNN(wv, M, K, x)
    ce = -np.dot(np.log(y.reshape(-1)), t.reshape(-1)) / N
    #ce=np.round(ce, 3)
    return ce


# test ---
WV = np.ones(15)
M = 2
K = 3
print(CE_FNN(WV, M, K, X_train[:2, :], T_train[:2, :]))
# 1.109
# 1.0986122886681098

# 리스트 7-1-(6)
# - 수치 미분 ------------------
def dCE_FNN_num(wv, M, K, x, t): # wv=WV; M; K; x=X_train[:2, :]; t=T_train[:2, :]
    epsilon = 0.001
    dwv = np.zeros_like(wv)
    for iwv in range(len(wv)):
        wv_modified = wv.copy()
        wv_modified[iwv] = wv[iwv] - epsilon
        mse1 = CE_FNN(wv_modified, M, K, x, t)
        wv_modified[iwv] = wv[iwv] + epsilon
        mse2 = CE_FNN(wv_modified, M, K, x, t)
        dwv[iwv] = (mse2 - mse1) / (2 * epsilon)
    return dwv


#--dVW의 표시 ------------------
def Show_WV(wv, M):
    N = wv.shape[0]
    plt.bar(range(1, M * 3 + 1), wv[:M * 3], align="center", color='black')
    plt.bar(range(M * 3 + 1, N + 1), wv[M * 3:],
            align="center", color='cornflowerblue')
    plt.xticks(range(1, N + 1))
    plt.xlim(0, N + 1)


#-test----
M = 2
K = 3
nWV = M * 3 + K * (M + 1)
np.random.seed(1)
WV = np.random.normal(0, 1, nWV)
dWV = dCE_FNN_num(WV, M, K, X_train[:2, :], T_train[:2, :])
print(dWV)
plt.figure(1, figsize=(5, 3))
Show_WV(dWV, M)
plt.show()

# 리스트 7-1-(7)
import time


# 수치 미분을 사용한 경사법 -------
def Fit_FNN_num(wv_init, M, K, x_train, t_train, x_test, t_test, n, alpha):
    # wv_init=WV_init; M; K; x_train=X_train; t_train=T_train
    # x_test=X_test; t_test=T_test; n=N_step; alpha
    wvt = wv_init
    err_train = np.zeros(n)
    err_test = np.zeros(n)
    wv_hist = np.zeros((n, len(wv_init)))
    epsilon = 0.001
    for i in range(n): # (A)
        wvt = wvt - alpha * dCE_FNN_num(wvt, M, K, x_train, t_train)
        err_train[i] = CE_FNN(wvt, M, K, x_train, t_train)
        err_test[i] = CE_FNN(wvt, M, K, x_test, t_test)
        wv_hist[i, :] = wvt
    return wvt, wv_hist, err_train, err_test


# 메인 ---------------------------
startTime = time.time()
M = 2
K = 3
np.random.seed(1)
WV_init = np.random.normal(0, 0.01, M * 3 + K * (M + 1))
N_step = 1000# (B) 학습 단계 1000
alpha = 0.5
WV, WV_hist, Err_train, Err_test = Fit_FNN_num(
    WV_init, M, K, X_train, T_train, X_test, T_test, N_step, alpha)
calculation_time = time.time() - startTime
print("Calculation time:{0:.3f} sec".format(calculation_time))

# 리스트 7-1-(8)
# 학습 오차의 표시 ---------------------------
plt.figure(1, figsize=(3, 3))
plt.plot(Err_train, 'black', label='training')
plt.plot(Err_test, 'cornflowerblue', label ='test')
plt.legend()
plt.show()

# 리스트 7-1-(9)
# 가중치의 시간 변화의 표시 ---------------------------
plt.figure(1, figsize=(3, 3))
plt.plot(WV_hist[:, :M * 3], 'black')
plt.plot(WV_hist[:, M * 3:], 'cornflowerblue')
plt.show()

'''''''''
WV_hist[0]=[ 0.01559798 -0.00655298 -0.00536427 -0.01041016  0.00958275 -0.02274073
  0.00812405 -0.01633896 -0.01475569  0.02524814  0.04249743  0.03523614
 -0.02164195 -0.02299    -0.02655379]

WV_hist[1]=[ 0.01512002 -0.00510388 -0.00497361 -0.00998806  0.01239252 -0.0220789
  0.00115777 -0.02279944 -0.02804574  0.04592615  0.06347334  0.07699673
 -0.03535368 -0.03750543 -0.05502435]

WV_hist[2]=[ 0.01469654 -0.00227775 -0.00443265 -0.00956753  0.01658928 -0.02133332
 -0.00394765 -0.02748081 -0.03757536  0.06124007  0.07922206  0.10789727
 -0.04556218 -0.04857279 -0.07639529]

WV_hist[3]= [ 0.01426978  0.00155209 -0.00389596 -0.00919871  0.02181073 -0.02064259
 -0.00766077 -0.03084358 -0.04421801  0.07257336  0.09110734  0.13058389
 -0.05318235 -0.0570953  -0.09243927]

WV_hist[4]= [ 0.01381431  0.00611537 -0.00342693 -0.00890074  0.02779576 -0.02005766
 -0.01037259 -0.03327257 -0.04871312  0.0810039   0.10019303  0.14712864
 -0.05890108 -0.063752   -0.10448893]

'''


# 리스트 7-1-(10)
# 경계선 표시 함수 --------------------------
def show_FNN(wv, M, K):
    xn = 60  # 등고선 표시 해상도
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    x = np.c_[np.reshape(xx0, xn * xn, 1), np.reshape(xx1, xn * xn, 1)]
    y, a, z, b = FNN(wv, M, K, x)
    plt.figure(1, figsize=(4, 4))
    for ic in range(K):
        f = y[:, ic]
        f = f.reshape(xn, xn)
        f = f.T
        cont = plt.contour(xx0, xx1, f, levels=[0.8, 0.9],
                           colors=['cornflowerblue', 'black'])
        cont.clabel(fmt='%1.1f', fontsize=9)
    plt.xlim(X_range0)
    plt.ylim(X_range1)


# 경계선 표시 --------------------------
plt.figure(1, figsize=(3, 3))
Show_data(X_test, T_test)
show_FNN(WV, M, K)
plt.show()

# 리스트 7-1-(14)
from mpl_toolkits.mplot3d import Axes3D


def show_activation3d(ax, v, v_ticks, title_str):
    f = v.copy()
    f = f.reshape(xn, xn)
    f = f.T
    ax.plot_surface(xx0, xx1, f, color='blue', edgecolor='black',
                    rstride=1, cstride=1, alpha=0.5)
    ax.view_init(70, -110)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticks(v_ticks)
    ax.set_title(title_str, fontsize=18)


M = 2
K = 3
xn = 15  # 등고선 표시 해상도
x0 = np.linspace(X_range0[0], X_range0[1], xn)
x1 = np.linspace(X_range1[0], X_range1[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
x = np.c_[np.reshape(xx0, xn * xn, 1), np.reshape(xx1, xn * xn, 1)]
y, a, z, b = FNN(WV, M, K, x)

fig = plt.figure(1, figsize=(12, 9))
plt.subplots_adjust(left=0.075, bottom=0.05, right=0.95,
                    top=0.95, wspace=0.4, hspace=0.4)

for m in range(M):
    ax = fig.add_subplot(3, 4, 1 + m * 4, projection='3d')
    show_activation3d(ax, b[:, m], [-10, 10], '$b_{0:d}$'.format(m))
    ax = fig.add_subplot(3, 4, 2 + m * 4, projection='3d')
    show_activation3d(ax, z[:, m], [0, 1], '$z_{0:d}$'.format(m))

for k in range(K):
    ax = fig.add_subplot(3, 4, 3 + k * 4, projection='3d')
    show_activation3d(ax, a[:, k], [-5, 5], '$a_{0:d}$'.format(k))
    ax = fig.add_subplot(3, 4, 4 + k * 4, projection='3d')
    show_activation3d(ax, y[:, k], [0, 1], '$y_{0:d}$'.format(k))

plt.show()
