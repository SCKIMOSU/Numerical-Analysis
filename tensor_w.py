"""
선형회귀 학습 — TensorFlow 없이 NumPy로 구현
원본 TF 1.x 코드를 동등하게 재구현하면서, GD 갱신을
수동으로 작성하여 학습 과정의 투명성을 확보.

수치해석 강의용
"""
import numpy as np
import matplotlib.pyplot as plt

# 재현성을 위한 시드 (결과를 일관되게 보여주려면 고정)
rng = np.random.default_rng(42)


# ============================================================
# 1. 데이터 생성
# ============================================================
def data_generation(num_points):
    """
    합성 데이터: y = 5x + noise

    x ~ N(2, 2²) + 10  →  대략 [10, 14] 범위에 68%
    y = 5x + 2·N(0, 3²) →  진짜 기울기는 5, 노이즈가 섞임
    """
    x_data = rng.normal(2, 2, num_points) + 10
    y_data = x_data * 5 + rng.normal(0, 3, num_points) * 2
    return x_data, y_data


# ============================================================
# 2. 데이터 시각화 (학습 전)
# ============================================================
def data_draw(x_data, y_data):
    plt.figure(figsize=(7, 5))
    plt.plot(x_data, y_data, 'ro', label='data')
    plt.xlim([0, 25])
    plt.ylim([0, 100])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Generated data (N = {len(x_data)})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# ============================================================
# 3. 경사하강법 학습 — NumPy로 직접
# ============================================================
def data_learning(x_data, y_data, lr=0.0015, n_steps=10, animate=True):
    """
    선형 모델 y = W·x + b 의 W, b를 GD로 학습.

    MSE 손실:           L = (1/N) · Σ (W·xᵢ + b - yᵢ)²
    그래디언트 (해석):  ∂L/∂W = (2/N) · Σ xᵢ·(W·xᵢ + b - yᵢ)
                       ∂L/∂b = (2/N) · Σ (W·xᵢ + b - yᵢ)
    """
    # ---- 초기화 (원본 TF 코드와 동등) ----
    W = rng.uniform(-1.0, 1.0)   # tf.Variable(tf.random_uniform([1], -1, 1))
    b = 0.0                       # tf.Variable(tf.zeros([1]))
    N = len(x_data)

    print(f'초기값:  W = {W:.6f},  b = {b:.6f}')
    print('-' * 60)

    # 학습 진행 애니메이션 셋업
    if animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))

    history = []

    for step in range(n_steps):
        # ---- Forward: 예측과 손실 ----
        y_pred = W * x_data + b
        residual = y_pred - y_data
        loss = np.mean(residual ** 2)

        # ---- Backward: 해석적 그래디언트 ----
        grad_W = 2.0 * np.mean(residual * x_data)
        grad_b = 2.0 * np.mean(residual)

        # ---- Update: 파라미터 갱신 ----
        W -= lr * grad_W
        b -= lr * grad_b

        history.append((step, W, b, loss))
        print(f'step {step:2d}   W = {W:8.4f}   b = {b:8.4f}   loss = {loss:8.4f}')

        # ---- 시각화: 같은 figure에 누적 ----
        if animate:
            ax.clear()
            ax.plot(x_data, y_data, 'ro', label='data', alpha=0.7)
            x_line = np.array([0, 25])
            ax.plot(x_line, W * x_line + b, 'b-', linewidth=2,
                    label=f'h(x) = {W:.3f}·x + {b:.3f}')
            ax.set_xlim([0, 25])
            ax.set_ylim([0, 100])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'step {step}:  loss = {loss:.4f}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            plt.pause(0.5)

    print('-' * 60)
    print(f'최종:    W = {W:.6f},  b = {b:.6f}')
    print(f'(참고: 진짜 기울기 ≈ 5, 진짜 절편 ≈ 0)')

    if animate:
        plt.ioff()
        plt.show()

    return W, b, history


# ============================================================
# 4. 메인
# ============================================================
if __name__ == '__main__':
    num_points = 50

    # (1) 데이터 생성
    x_data, y_data = data_generation(num_points)

    # (2) 학습 전 데이터 확인
    data_draw(x_data, y_data)

    # (3) 경사하강법 학습
    W_final, b_final, history = data_learning(
        x_data, y_data,
        lr=0.0015,    # 적절한 학습률 (10 step으로 수렴)
        n_steps=10,
        animate=True
    )
