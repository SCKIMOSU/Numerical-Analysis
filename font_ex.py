import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as	np
font1={'family':'NanumMyeongjo','color':'black','size':	24}
font2={'family':'NanumBarunpen','color':'darkred','weight':'bold', 'size':18}
font3={'family':'NanumBarunGothic','color':'blue','weight':'light','size':12}
x=np.linspace(0.0,5.0,100)
y=np.cos(2*np.pi*x)*np.exp(-x)
plt.plot(x,y,'k')
plt.title('한글 제목', fontdict=font1)
plt.xlabel('엑스 축', fontdict=font2)
plt.ylabel('와이 축', fontdict=font3)
plt.subplots_adjust()
plt.show()
