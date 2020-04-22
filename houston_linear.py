# https://www.mlgorithm.com/blog/linear-regression-least-square-method
# 집값과 연봉의 관련성은?
# 개인의 연봉 수입과 소유 집값에는 어떤 연관성이 있을까?
# 수입이 높으면, 비싼 집에 사는가?
# 미국 휴스톤의 예를 보자.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline
# df = pd.read_csv(r'C:\Users\Richie Rich\Desktop\Data Files\USA_Housing.csv')
# df = pd.read_csv(r'C:\Users\sckMac\PycharmProjects\class1\USA_Housing.csv')
#df = pd.read_csv(r'/home/sckubuntu/다운로드/USA_Housing.csv')
#df = pd.read_csv(r'USA_Housing.csv')
#data = pd.read_csv('file1.csv', error_bad_lines=False)

#df = pd.read_csv(r'USA_Housing.csv', error_bad_lines=False)
#data=pd.read_csv("File_path", sep='\t')

df=pd.read_csv(r'USA_Housing.csv', sep='\t')

df.info()
df.describe()
df.head(10)

# as we don't want address column so we will drop it
df = df.drop('Address',axis=1)
X = df.drop('Price',axis=1)

X1 = df['Avg. Area Income']

# as in x we will every feature expect price
Y = df['Price']

p1=np.polyfit(X1, Y , 1)
# array([ 2.11954832e+01, -2.21579478e+05])
p2=np.polyfit(X1, Y, 2)
#array([4.98015093e-05, 1.43822894e+01, 5.78606281e+03])
p3=np.polyfit(X1, Y, 3)
# array([ 3.32345038e-10, -1.79963602e-05,  1.88774722e+01, -9.08744499e+04])

plt.figure(1)
plt.plot(X1, Y, 'o')
#plt.plot(x,fx, 'r*-')

plt.figure(2)
plt.plot(X1, Y, 'o')
plt.plot(X1, np.polyval(p1, X1), 'b*-')

plt.figure(3)
plt.plot(X1, Y, 'o')
plt.plot(X1, np.polyval(p1,X1), 'r*-')
plt.plot(X1, np.polyval(p2,X1), 'g>-')
plt.plot(X1, np.polyval(p3,X1), 'mx-')
# 여기까지만 

# now we have to do train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.4,random_state=101)

# now we have to import linear regression model
from sklearn.linear_model import LinearRegression
lm = LinearRegression() # instantiating the linear model
lm.fit(X_train,Y_train)
print(lm.intercept_)
print(lm.coef_) # coef will relate to each feature in our data set
cdf=pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])
cdf.head()

# It means if all the other features are fixed then 1 unit increase in Avg.Area income is associated with increase of 21.52$ in price
#  now we can do prediction
predictions = lm.predict(X_test)
predictions # contains predicted values
Y_test # contains real values
# let's see the result on scatter plot
plt.figure(1)
plt.scatter(Y_test, predictions)
plt.show()

# As we are getting a straight line which means we did pretty good job.
# Let's create a histogram :
plt.figure(2)
sns.distplot((Y_test-predictions))
plt.show()

from sklearn import metrics
np.sqrt(metrics.mean_squared_error(Y_test, predictions))

