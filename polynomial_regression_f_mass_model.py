import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import platform

# 한글 폰트 설정
def set_korean_font():
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()
# 상수 정의
g = 9.81
cd = 0.25
t = 4

# 속도 함수
def v_of_m(m):
    return np.sqrt(g * m / cd) * np.tanh(np.sqrt(g * cd / m) * t)

# 1. 데이터 생성
m_values = np.linspace(30, 200, 300)
v_values = v_of_m(m_values)
v_diff = v_values - 36  # 기준 속도 36과의 차이

# 2. 다항 회귀 모델 피팅 (3차)
coeffs = np.polyfit(m_values, v_diff, deg=3)
poly_model = np.poly1d(coeffs)
v_fit = poly_model(m_values)

# 3. 그래프 출력
plt.figure(figsize=(10, 6))
plt.plot(m_values, v_diff, 'b-', label='실제 v(m) - 36')
plt.plot(m_values, v_fit, 'r--', label='3차 다항 근사 모델')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('몸무게 m (kg)')
plt.ylabel('속도 - 36 (m/s)')
plt.title('몸무게에 따른 (속도 - 36)의 변화 및 근사 모델')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. 회귀 모델 수식 출력
print("\n✅ 다항 회귀 모델 식 (v(m) - 36 ≈ ...):")
print(poly_model)
