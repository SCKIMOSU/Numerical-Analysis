import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 폰트 설정 (선택)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터
x = np.array([10, 11, 12, 13, 14])
t = np.array([135, 140, 145, 150, 155])

# 오차 함수 미분
def dmse_line(x, t, w):
    y = w[0] * x + w[1]
    d_w0 = 2 * np.mean((y - t) * x)
    d_w1 = 2 * np.mean(y - t)
    return d_w0, d_w1

# 학습
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

# 🔧 애니메이션 그래프 생성
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 왼쪽: 예측 선
line, = ax1.plot([], [], 'b-', label='모델 예측')
scatter = ax1.scatter(x, t, color='black', label='실제 데이터')
text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes)
ax1.set_xlim(9, 15)
ax1.set_ylim(130, 160)
ax1.set_xlabel('나이 (x)')
ax1.set_ylabel('키 (y)')
ax1.set_title('경사하강법 학습 예측')
ax1.legend()
ax1.grid(True)

# 오른쪽: 손실 그래프
loss_line, = ax2.plot([], [], 'r-o', label='MSE')
ax2.set_xlim(0, epochs)
ax2.set_ylim(0, max(loss_history) * 1.1)
ax2.set_xlabel('학습 단계 (epoch)')
ax2.set_ylabel('손실 값 (MSE)')
ax2.set_title('손실 함수 감소')
ax2.grid(True)
ax2.legend()

def init():
    line.set_data([], [])
    loss_line.set_data([], [])
    text.set_text('')
    return line, loss_line, text

def animate(i):
    w0, w1 = W_history[i]
    y_pred = w0 * x + w1
    line.set_data(x, y_pred)

    loss_line.set_data(range(i + 1), loss_history[:i + 1])
    text.set_text(f"Step {i+1}: w0={w0:.2f}, w1={w1:.2f}, loss={loss_history[i]:.2f}")
    return line, loss_line, text

ani = animation.FuncAnimation(
    fig, animate, frames=len(W_history),
    init_func=init, blit=True, interval=300, repeat=False
)

# 저장
ani.save("gradient_descent_with_loss.gif", writer='pillow', fps=3)
print("✅ gradient_descent_with_loss.gif 저장 완료")
