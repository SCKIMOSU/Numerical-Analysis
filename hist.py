import numpy as np
import matplotlib.pyplot as plt

# plt.hist() 로 히스토그램 그리기
s=np.random.uniform(6,7,25)
plt.figure(4)
plt.hist(s,5)
plt.figure(5)
plt.hist(s,4, color='g')

# plt.bar()히스토그램 그리기
smin=s.min()
smax=s.max()
bin_width=(s.max()-s.min())/4
first_left=smin
second_left=smin+bin_width
third_left=smin+bin_width*2
fourth_left=smin+bin_width*3
fifth_left=smin+bin_width*4

bins_left=np.array([first_left, second_left, third_left, fourth_left ])

first_height=np.size(np.where( (s < second_left) & ( s>= first_left) ))
second_height=np.size(np.where( (s < third_left) & ( s>= second_left) ))
third_height=np.size(np.where( (s < fourth_left) & ( s>= third_left) ))
fourth_height=np.size(np.where( (s <= fifth_left) & ( s>= fourth_left) ))
bins_height=np.array([first_height, second_height, third_height, fourth_height])

plt.figure(6)
plt.subplot(2,1,1)
plt.bar(bins_left, bins_height, width = bin_width)\
# 콘솔 디버깅 (plt.bar(x) 시 주의 사항 )
# plt.bar(x, height, width)
# x=bins_left 를 의미함  *****
# height=bins_height 를 의미함
# width=bin_width 를 의미함

plt.grid()

plt.subplot(2,1,2)
plt.bar(bins_left, bins_height, width = bin_width-0.05)
plt.grid()
plt.show()

# plt.bar()와 plt.hist()를 이용한 히스토그램
plt.figure(7)
plt.subplot(2, 1, 1)
plt.bar(bins_left, bins_height, bin_width)
plt.title('Bar')
plt.ylabel('Count')
plt.grid()
plt.subplot(2, 1, 2)
plt.hist(s, bins=4, color='c', alpha=0.75)
plt.title('Histrogram')
plt.ylabel('Count')
plt.grid()

# 이산신호: Probability Mass Function (pmf: 확률질량함수)
# s : 25개는 이산신호로 pmf를 사용하는 것이 맞음
# 연속신호: Probability Density Function (pdf: 확률밀도함수)
# 그러나, s 가 무한대의 수로 가게 되면 이산신호가
# 아니고 연속신호가 됨으로 pdf를 사용함
# 수치해석에서는 이산/연속 구분하지 않고 pdf를 사용해서
# cdf로 연관시킬 예정임

pdf=bins_height/np.size(s)
plt.figure(8)
plt.subplot(2, 1, 1)
plt.hist(s, bins=4, color='c', alpha=0.75)
plt.title('Histrogram')
plt.ylabel('Count')
plt.grid()
plt.subplot(2, 1, 2)
plt.bar(bins_left, pdf, bin_width)
plt.title('Bar')
plt.ylabel('pdf')
plt.grid()


plt.figure(9)
plt.plot(bins_left, pdf, 'b*-')
plt.title('pdf')
plt.xlabel('s')
plt.ylabel('pdf')
plt.grid()
plt.show()

# Cumulative Density Function (누적 분포함수)
pdf=bins_height/np.size(s)
cdf=np.cumsum(pdf)
plt.figure(10)
#plt.subplot(1, 2, 2)
plt.plot(bins_left, cdf, 'ro-')
plt.title('cdf')
plt.xlabel('s')
plt.ylabel('cdf')
plt.grid()
plt.show()


# 더 많은 데이터 셋에 대해 확률 밀도 함수를 적용
x3=np.random.randn(10000)
plt.figure(11)
plt.subplot(2, 1, 1)
n3, bins3, patches3 = plt.hist(x3, bins=50, color='b', alpha=0.75)
plt.title('Histrogram')
plt.ylabel('Count')
plt.subplot(2, 1, 2)
n4, bins4, patches4 = plt.hist(x3, 50, normed=1, facecolor='b', alpha=0.75)
plt.xlabel('Probability Density Function')
plt.ylabel('Probability')


# 더 많은 데이터 셋에 대해 plt.bar()함수로 확률 밀도 함수 구현
nd=np.random.randn(10000)
ra=np.int(np.floor(np.min(nd)))
rb=np.int(np.ceil(np.max(nd)))
hist, bin_left=np.histogram(nd, bins=np.arange(ra, rb+1, 1))
bin_width=(rb-ra)/np.size(bin_left)
pdf=hist/np.size(nd)
cdf=np.cumsum(pdf)
plt.figure(12)
plt.subplot(1, 2, 1)
plt.bar(bin_left[:-1], pdf, bin_width)
plt.title('pdf')
plt.xlabel('normal distribution')
plt.ylabel('pdf')

plt.subplot(1, 2, 2)
plt.bar(bin_left[:-1], cdf, bin_width, facecolor='m')
plt.title('cdf')
plt.xlabel('normal distribution')
plt.ylabel('cdf')
plt.show()

#plt.plot()으로 cdf시각화
plt.figure(13)
#plt.subplot(1, 2, 2)
plt.plot(bin_left[:-1], cdf, 'ro-')
plt.title('cdf')
plt.xlabel('normal distribution')
plt.ylabel('cdf')
plt.grid()
plt.show()

#plt.semilogy()로 cdf시각화
plt.figure(14)
#plt.subplot(1, 2, 2)
plt.semilogy(bin_left[:-1], cdf, 'b*-')
plt.title('cdf')
plt.xlabel('normal distribution')
plt.ylabel('cdf')
plt.grid()
plt.show()
