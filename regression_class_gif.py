import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import imageio
import os

def data_generation(num_points=50):
    vectors_set = []
    for _ in range(num_points):
        x = np.random.normal(2, 2) + 10
        y = x * 5 + (np.random.normal(0, 3)) * 2
        vectors_set.append([x, y])
    x_data = np.array([v[0] for v in vectors_set], dtype=np.float32)
    y_data = np.array([v[1] for v in vectors_set], dtype=np.float32)
    return x_data, y_data

def data_learning_with_gif(x_data, y_data, gif_filename="linear_regression.gif"):
    W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    learning_rate = 0.0015
    frames = []

    # 결과 저장용 디렉토리 생성
    os.makedirs("frames", exist_ok=True)

    for step in range(20):
        with tf.GradientTape() as tape:
            y_pred = W * x_data + b
            loss = tf.reduce_mean(tf.square(y_pred - y_data))

        gradients = tape.gradient(loss, [W, b])
        W.assign_sub(learning_rate * gradients[0])
        b.assign_sub(learning_rate * gradients[1])

        print(f"Step {step:02d}: W = {W.numpy()}, b = {b.numpy()}, Loss = {loss.numpy():.4f}")

        # 시각화 후 이미지 저장
        plt.figure()
        plt.plot(x_data, y_data, 'ro', label='Data')
        plt.plot(x_data, (W * x_data + b).numpy(), 'b-', label=f'Step {step}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim([0, 100])
        plt.xlim([0, 25])
        plt.legend()
        plt.title(f"Step {step}, Loss: {loss.numpy():.2f}")
        filename = f"frames/frame_{step:02d}.png"
        plt.savefig(filename)
        frames.append(imageio.imread(filename))
        plt.close()

    # GIF로 저장
    imageio.mimsave(gif_filename, frames, duration=0.5)
    print(f"\n✅ GIF saved as '{gif_filename}'")

    # 임시 이미지 삭제
    for f in os.listdir("frames"):
        os.remove(os.path.join("frames", f))
    os.rmdir("frames")

if __name__ == '__main__':
    x_data, y_data = data_generation()
    data_learning_with_gif(x_data, y_data)
