#출처: https://rfriend.tistory.com/411 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]

# importing packages

import numpy as np

import pandas as pd


import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['figure.figsize'] = [10, 6]



# Loading 'tips' dataset from seaborn

tips = sns.load_dataset('tips')

tips.shape

tips.head()


# Summary Statistics

tips_sum_by_day = tips.groupby('day').tip.sum()

tips_sum_by_day

label = ['Thur', 'Fri', 'Sat', 'Sun']

index = np.arange(len(label))

# Basic Bar Chart

plt.bar(index, tips_sum_by_day)

plt.title('Sum of Tips by Day', fontsize=20)

plt.xlabel('Day', fontsize=18)

plt.ylabel('Sum of Tips', fontsize=18)

plt.xticks(index, label, fontsize=15)

plt.show()

# bar color, transparency

plt.bar(label, tips_sum_by_day,

        color='red', # color

        alpha=0.5) # transparency

plt.show()

# bar width, align

plt.bar(label, tips_sum_by_day,

        width=0.5, # default: 0.8

        align='edge') # default: 'center'

plt.show()



# X tick labels rotation

plt.bar(index, tips_sum_by_day)

plt.xlabel('Day', fontsize=18)

plt.xticks(index, label, fontsize=15,

           rotation=90) # when X tick labels are long

plt.show()



# Horizontal Bar Chart

plt.barh(index, tips_sum_by_day)

plt.title('Sum of Tips by Day', fontsize=18)

plt.ylabel('Day', fontsize=15)

plt.xlabel('Sum of Tips', fontsize=15)

plt.yticks(index, label, fontsize=13, rotation=0)

plt.show()











