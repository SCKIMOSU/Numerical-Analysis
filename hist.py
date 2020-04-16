import numpy as np
import matplotlib.pyplot as plt

# plt.hist() 로 히스토그램 그리기
s=np.random.uniform(6,7,25)
plt.figure(4)
plt.hist(s,5)
plt.figure(5)
plt.hist(s,4, color='g')

# bar() 히스토그램 그리기
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
fourth_height=np.size(np.where( (s < fifth_left) & ( s>= fourth_left) ))
bins_height=np.array([first_height, second_height, third_height, fourth_height])

plt.figure(6)
plt.subplot(2,1,1)
plt.bar(bins_left, bins_height, width = bin_width)
plt.grid()

plt.subplot(2,1,2)
plt.bar(bins_left, bins_height, width = bin_width-0.05)
plt.grid()
plt.show()

