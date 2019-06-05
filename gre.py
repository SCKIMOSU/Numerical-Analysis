import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
# 출처: https://3months.tistory.com/27?category=753896 [Deep Play]

df = pd.read_csv("binary.csv")
#df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
print (df.head())

df.columns = ["admit", "gre", "gpa", "prestige"]  # df의 column 이름 바꾸기
print (df.columns)

print (df.describe())
  # 빈도수, 평균, 분산, 최솟값, 최댓값, 1/4분위수, 중위값, 1/4분위수를 나타냄

#             admit         gre         gpa   prestige
# count  400.000000  400.000000  400.000000  400.00000
# mean     0.317500  587.700000    3.389900    2.48500
# std      0.466087  115.516536    0.380567    0.94446
# min      0.000000  220.000000    2.260000    1.00000
# 25%      0.000000  520.000000    3.130000    2.00000
# 50%      0.000000  580.000000    3.395000    2.00000
# 75%      1.000000  660.000000    3.670000    3.00000
# max      1.000000  800.000000    4.000000    4.00000

print (df.std())
  # 분산 출력

# admit      0.466087
# gre      115.516536
# gpa        0.380567
# prestige   0.944460

print (pd.crosstab(df['admit'], df['prestige'], rownames=['admit']))


# prestige   1   2   3   4
# admit
# 0         28  97  93  55
# 1         33  54  28  12

df.hist()
pl.show()  # pl.show()를 해야 화면에 띄워준다! 결과는 아래와 같다. 모든 컬럼에 대해 히스토그램을 그림

dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
print (dummy_ranks.head())


#    prestige_1  prestige_2  prestige_3  prestige_4
# 0           0           0           1           0
# 1           0           0           1           0
# 2           1           0           0           0
# 3           0           0           0           1
# 4           0           0           0           1

cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
print (data.head())

#    admit  gre   gpa  prestige_2  prestige_3  prestige_4
# 0      0  380  3.61           0           1           0
# 1      1  660  3.67           0           1           0
# 2      1  800  4.00           0           0           0
# 3      1  640  3.19           0           0           1
# 4      0  520  2.93           0           0           1

data['intercept'] = 1.0


train_cols = data.columns[1:]
logit = sm.Logit(data['admit'], data[train_cols])
result = logit.fit()
print (result.summary())

#coef에 주목한다. gre:0.0023 gpa :0.840, prestige_2 : -0.6754 등등...
#coef(편회귀계수)의 값이 양수이면 그 컬럼의 값이 커질수록 목적변수가 TRUE일 확률 즉, admit=1일 확률이 높아진다.
#반대로 coef의 값이 음수이면 그 컬럼의 값이 커질수록 목적변수가 FALSE일 확률 즉, admin=0일 확률이 높아진다.

#즉 GRE나 GPA가 커질수록 대학원에 입학할 확률은 커지고 prestige_2, prestige_3이 커질수록 대학원에 입학할 확률은 작아진다.
#이러한 경향은 pretige가 낮아질수록 심해진다.




print (np.exp(result.params))


