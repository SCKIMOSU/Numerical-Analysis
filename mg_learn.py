import numpy as np
import matplotlib.pyplot as plt
import mglearn
# %matplotlib inline

plt.figure(1)
mglearn.plots.plot_knn_classification(n_neighbors=1)

plt.figure(2)
mglearn.plots.plot_knn_classification(n_neighbors=3)

# https://minwoo2815.tistory.com/entry/Scikit-Learn-%EC%A7%80%EB%8F%84%ED%95%99%EC%8A%B5-K-%EC%B5%9C%EA%B7%BC%EC%A0%91%EC%9D%B4%EC%9B%83-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
from sklearn.model_selection import train_test_split
# mglearn의 make_forge 함수를 이용하여서 data와 target을 x와 y에 대입하겠습니다.
x, y = mglearn.datasets.make_forge()

# 학습된 모델(KNN)이 잘 학습되었고,
# 새로운 데이터에도 잘 예측해내는지 알아보기 위해서
# 훈련셋, 테스트셋(새로운 데이터)으로 나누었습니다.
# 각각 저 데이터셋의 75%, 25%가 x_train y_train , x_test y_test에 담겨있습니다.

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0 )

plt.figure(3)
mglearn.discrete_scatter(x[:, 0], x[:, 1], y)

from sklearn.neighbors import KNeighborsClassifier

# KNN알고리즘을 import해서 찾을 최근접 이웃 갯수를 3개로 설정한뒤,
# fit을 하여서 학습데이터로 모델을 학습시킵니다.
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(x_train, y_train)

# 예측을 해보니 이런 결과가 나왔습니다. 예측률을 한번 보죠
print("테스트 데이터 세트 예측 : \n{}".format(knn_classifier.predict(x_test)))

# 정확도는 86%가 나왔습니다. 이말은 모델이 테스트 데이터셋에 있는 데이터들중 86%를 제대로
# 맞췄다는 말입니다.
print("테스트 데이터 세트 정확도 : \n{}".format(knn_classifier.score(x_test, y_test)))







