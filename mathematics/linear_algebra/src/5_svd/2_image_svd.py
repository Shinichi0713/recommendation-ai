import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ランク1の2x2行列を例として作成
# 例: A = [[1, 0.5], [2, 1]] はランク1（2行が線形従属）
A = np.array([[1.0, 0.5],
              [2.0, 1.0]])

# SVDを計算
U, S, Vt = np.linalg.svd(A)
print("A =")
print(A)
print("\nU =")
print(U)
print("\nS =", S)
print("\nVt =")
print(Vt)

# 特異値の個数 = ランク r
r = np.sum(S > 1e-10)  # 数値誤差を考慮
print(f"\nrank(A) = {r}")

# 左特異ベクトル
u1 = U[:, 0]
u2 = U[:, 1]

print(f"\nu1 = {u1}  (σ1 = {S[0]})")
print(f"u2 = {u2}  (σ2 = {S[1]})")

# 単位円上の点をサンプリング
theta = np.linspace(0, 2*np.pi, 100)
x_circle = np.cos(theta)
y_circle = np.sin(theta)
circle_points = np.vstack([x_circle, y_circle])

# Aで写した像
image_points = A @ circle_points

# 可視化
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('Image of A and left singular vectors')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# 単位円（入力）を薄く描画
ax.plot(x_circle, y_circle, 'gray', linestyle='--', alpha=0.5, label='Unit circle (input)')

# Aの像（出力）を描画
ax.plot(image_points[0], image_points[1], 'b-', label='A(unit circle)')

# 左特異ベクトルを矢印で描画
scale = 2.0
ax.arrow(0, 0, scale*u1[0], scale*u1[1], head_width=0.1, head_length=0.1, fc='red', ec='red', label=f'u1 (σ1={S[0]:.3f})')
ax.arrow(0, 0, scale*u2[0], scale*u2[1], head_width=0.1, head_length=0.1, fc='orange', ec='orange', label=f'u2 (σ2={S[1]:.3f})')

# u1, u2 が張る直線を描画（像空間の「軸」）
t = np.linspace(-3, 3, 100)
line_u1 = np.outer(u1, t)  # u1 方向の直線
line_u2 = np.outer(u2, t)  # u2 方向の直線

ax.plot(line_u1[0], line_u1[1], 'r--', alpha=0.7, label='span{u1}')
ax.plot(line_u2[0], line_u2[1], 'orange', linestyle='--', alpha=0.7, label='span{u2}')

ax.legend()
plt.show()

# アニメーションで「入力ベクトル → 像」の対応を見る（オプション）
fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
ax_anim.set_xlim(-3, 3)
ax_anim.set_ylim(-3, 3)
ax_anim.set_aspect('equal')
ax_anim.grid(True)
ax_anim.set_title('Input vector → Image under A')
ax_anim.set_xlabel('x1')
ax_anim.set_ylabel('x2')

# u1, u2 方向の直線を描画
ax_anim.plot(line_u1[0], line_u1[1], 'r--', alpha=0.7, label='span{u1}')
ax_anim.plot(line_u2[0], line_u2[1], 'orange', linestyle='--', alpha=0.7, label='span{u2}')

# アニメーション用の点
point_input, = ax_anim.plot([], [], 'go', markersize=8, label='Input vector')
point_image, = ax_anim.plot([], [], 'bo', markersize=8, label='Image under A')
ax_anim.legend()

def animate(frame):
    # frame に対応する角度
    theta_frame = theta[frame]
    x_in = np.cos(theta_frame)
    y_in = np.sin(theta_frame)
    point_input.set_data([x_in], [y_in])
    
    # Aで写した点
    x_out, y_out = A @ np.array([x_in, y_in])
    point_image.set_data([x_out], [y_out])
    
    return point_input, point_image

anim = FuncAnimation(fig_anim, animate, frames=len(theta), interval=50, blit=True)
plt.show()