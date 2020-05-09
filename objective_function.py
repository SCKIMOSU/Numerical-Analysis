import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

font1={'family':'NanumMyeongjo','color':'black','size':	24}
font2={'family':'NanumBarunpen','color':'darkred','weight':'bold', 'size':18}
font3={'family':'NanumBarunGothic','color':'blue','weight':'light','size':12}

#x=np.linspace(0.0,5.0,100)
#y=np.cos(2*np.pi*x)*np.exp(-x)
#plt.plot(x,y,'k')
#plt.title('한글 제목', fontdict=font1)
#plt.xlabel('엑스 축', fontdict=font2)
#plt.ylabel('와이 축', fontdict=font3)
#plt.subplots_adjust()
#plt.show()

# 다음은 1차원 목적함수의 예이다.
# 그래프에서 이 목적함수  f1(x) 의 최저점은  x∗=2 임을 알 수 있다.
# https://datascienceschool.net/view-notebook/4642b9f187784444b8f3a8309c583007/

def f1(x):
    return (x - 2) ** 2 + 2

xx = np.linspace(-1, 4, 100)
plt.plot(xx, f1(xx))
plt.plot(2, 2, 'ro', markersize=10)
plt.ylim(0, 10)
plt.xlabel("x")
plt.ylabel("$f_1(x)$")
plt.title("1차원 목적함수", fontdict=font3)
plt.show()


def f2(x, y):
    return (1 - x)**2 + 100.0 * (y - x**2)**2

xx = np.linspace(-4, 4, 800)
yy = np.linspace(-3, 3, 600)
X, Y = np.meshgrid(xx, yy)
Z = f2(X, Y)

levels=np.logspace(-1, 3, 10)
plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
plt.contour(X, Y, Z, colors="gray",
            levels=[0.4, 3, 15, 50, 150, 500, 1500, 5000])
plt.plot(1, 1, 'ro', markersize=10)

plt.xlim(-4, 4)
plt.ylim(-3, 3)
plt.xticks(np.linspace(-4, 4, 9))
plt.yticks(np.linspace(-3, 3, 7))
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("2차원 로젠브록 함수 $f(x,y)$", fontdict=font3)
plt.show()

# https://frhyme.github.io/python-lib/python_contour/

Xmesh, Ymesh = np.meshgrid(np.linspace(-3.0, 3.0, 1000),
                     np.linspace(-3.0, 3.0, 1000)
                    )
# levels = np.linspace(Z.reshape(-1, 1).min(), Z.reshape(-1, 1).max(), 50)
# np.size(levels) = 50
print("XX.shape: {}".format(Xmesh.shape))
print("YY.shape: {}".format(Ymesh.shape))
Z = np.sqrt(Xmesh**2 + Ymesh**2 )

plt.figure(figsize=(12, 5))
"""levels에 구간을 넣어줘서 등고선 표시 위치를 정할 수 있습니다. 
"""
cp = plt.contourf(Xmesh, Ymesh, Z,
                 levels = np.linspace(Z.reshape(-1, 1).min(), Z.reshape(-1, 1).max(), 50)
                )
plt.colorbar(cp)
#plt.savefig('../../assets/images/markdown_img/draw_contour_20180529_1727.svg')
plt.savefig('draw_contour.svg')
plt.show()

