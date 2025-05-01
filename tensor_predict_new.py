import tensorflow as tf
import numpy as np

# 데이터 준비
x_data = np.array([1, 2, 3], dtype=np.float32)
y_data = np.array([1, 2, 3], dtype=np.float32)

# 변수 선언 (학습 대상)
W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random.uniform([1], -1.0, 1.0))

# 학습률 설정
learning_rate = 0.1

# 손실 함수 정의
def compute_loss():
    y_pred = W * x_data + b
    return tf.reduce_mean(tf.square(y_pred - y_data))

# 학습 함수 정의 (자동 미분 사용)
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss = compute_loss()
    gradients = tape.gradient(loss, [W, b])
    W.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])
    return loss

# 학습 루프 실행
for step in range(100):
    loss_val = train_step()
    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss_val.numpy():.6f}, W = {W.numpy()}, b = {b.numpy()}")

# 예측 결과 확인
print("\n=== Test ===")
print("X: 5 => Y:", (W * 5 + b).numpy())
print("X: 2.5 => Y:", (W * 2.5 + b).numpy())
