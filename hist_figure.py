# https://frhyme.github.io/python-lib/plt_hist/
import numpy as np
import matplotlib.pyplot as plt
## normal분포로 나온 값을 그대로 plt.hist에 넣어줍니다.
x = np.random.normal(0, 1, 100)

plt.figure(figsize=(10, 6))
## histogram의 경우 내가 값 리스트를 넣고, 입력한 bin 개수에 따라 알아서 분류해줌
## ys: y값,
## xs: x 값
## patches 일단, 필요없듬.
ys, xs, patches = plt.hist(x,
         bins=5, ## 몇 개의 바구니로 구분할 것인가.
         density=True, ## ytick을 퍼센트비율로 표현해줌
         cumulative=False, ## 누적으로 표현하고 싶을 때는 True
         histtype='bar',  ## 타입. or step으로 하면 모양이 바뀜.
         orientation='vertical', ## or horizontal
         rwidth=0.8, ## 1.0일 경우, 꽉 채움 작아질수록 간격이 생김
         color='hotpink', ## bar 색깔
        )

## figure에 bar의 개별 퍼센트를 보여주고 싶을 때.
for i in range(0, len(ys)):
    ## 앞에서 plt.hist 가 리턴하는 값이 bar의 x, y좌표이기 때문에
    ## 이 값을 이용해서 글자를 어디에 넣을지 결정해줌.
    plt.text(x=xs[i]+0.23, y=ys[i]+0.015,
             s='{:0>4.1f}%'.format(ys[i]*100), ## 넣을 스트링
             #fontproperties=BMJUA, ## 문자
             fontsize=20,## 크기
             ## 이유는 모르겠지만, fontsize를 fontproperties보다 먼저 설정하면 값이 안 먹힘
             color='red',)
y_min, y_max = plt.ylim() ## 글자가 안 보이는 경우가 있어서, 위의 길이를 조금 늘려줌
plt.ylim(y_min, y_max+0.05)

plt.yticks([])## text로 표시하고 있기 때문에 이 부분은 삭제해줌
## xticks을 변경해줌
plt.xticks([(xs[i]+xs[i+1])/2 for i in range(0, len(xs)-1)],
           ["{:.1f} ~ {:.1f}".format(xs[i], xs[i+1]) for i in range(0, len(xs)-1)])
#plt.savefig('../../assets/images/markdown_img/180802_plt_histogram.svg')
plt.savefig('plt_histogram.svg')
plt.show()
