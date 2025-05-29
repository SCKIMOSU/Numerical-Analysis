import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서도 작동하도록 설정
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 한글 폰트 설정 (선택)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터: 나이(x) → 키(t)
x = np.array([10, 11, 12, 13, 14])
t = np.array([135, 140, 145, 150, 155])

# 오차 함수 기울기 함수
def dmse_line(x, t, w):
    y = w[0] * x + w[1]
    d_w0 = 2 * np.mean((y - t) * x)
    d_w1 = 2 * np.mean(y - t)
    return d_w0, d_w1

# 초기값 및 경사하강법
w = np.array([0.0, 0.0])
eta = 0.001
epochs = 50
W_history = []
loss_history = []

for _ in range(epochs):
    d_w0, d_w1 = dmse_line(x, t, w)
    w[0] -= eta * d_w0
    w[1] -= eta * d_w1
    W_history.append(w.copy())

    y_pred = w[0] * x + w[1]
    loss = np.mean((y_pred - t) ** 2)
    loss_history.append(loss)

W_history = np.array(W_history)

# 애니메이션 시각화
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label='모델 예측')
scatter = ax.scatter(x, t, color='black', label='실제 데이터')
text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
ax.set_xlim(9, 15)
ax.set_ylim(130, 160)
ax.set_xlabel('나이 (x)')
ax.set_ylabel('키 (y)')
ax.set_title('경사하강법 애니메이션')
ax.legend()
ax.grid(True)

def init():
    line.set_data([], [])
    text.set_text('')
    return line, text

def animate(i):
    w0, w1 = W_history[i]
    y_pred = w0 * x + w1
    line.set_data(x, y_pred)
    text.set_text(f"Step {i+1}: w0={w0:.2f}, w1={w1:.2f}")
    return line, text

ani = animation.FuncAnimation(fig, animate, frames=len(W_history),
                              init_func=init, blit=True, interval=300, repeat=False)

# 🔽 🔽 🔽 파일로 저장 (GIF 형식)
ani.save("gradient_descent.gif", writer='pillow', fps=3)
print("✅ gradient_descent.gif 파일이 저장되었습니다.")
