import numpy as np
import matplotlib.pyplot as plt

# 재현성을 위한 시드 (선택)
rng = np.random.default_rng(42)


def data_generation(num_points):
    """y = 5x + noise 형태의 합성 데이터 생성"""
    x_data = rng.normal(2, 2, num_points) + 10           # x ~ N(12, 2²)
    y_data = x_data * 5 + rng.normal(0, 3, num_points) * 2  # y = 5x + noise
    return x_data, y_data


def data_draw(x_data, y_data):
    """데이터 산점도"""
    plt.plot(x_data, y_data, 'ro')
    plt.xlim([0, 25])
    plt.ylim([0, 100])
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()


def data_learning(x_data, y_data, lr=0.0015, n_steps=10):
    """경사하강법으로 W, b 학습"""
    # 초기화 — TF 코드와 동등하게 W는 [-1,1] 균등, b는 0
    W = rng.uniform(-1.0, 1.0)
    b = 0.0
    N = len(x_data)

    for step in range(n_steps):
        # Forward: 예측과 손실
        y_pred = W * x_data + b
        residual = y_pred - y_data
        loss = np.mean(residual ** 2)

        # Backward: 해석적 그래디언트
        grad_W = 2.0 * np.mean(residual * x_data)
        grad_b = 2.0 * np.mean(residual)

        # Update: 파라미터 갱신
        W -= lr * grad_W
        b -= lr * grad_b

        # 로그 + 시각화
        print(f'{step}  W=[{W:.6f}]  b=[{b:.6f}]')
        print(f'{step}  loss={loss:.4f}')

        plt.plot(x_data, y_data, 'ro')
        plt.plot(x_data, W * x_data + b)
        plt.xlabel('x'); plt.ylabel('y')
        plt.show()

    return W, b


if __name__ == '__main__':
    num_points = 50
    x_data, y_data = data_generation(num_points)
    data_draw(x_data, y_data)
    data_learning(x_data, y_data)
