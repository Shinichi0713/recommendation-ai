import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

# 任意の2x2行列 A（ここでは例として適当な行列を設定）
A = np.array([[1.2, 0.8],
              [0.5, 1.0]])

# SVD を計算
U, S, Vt = np.linalg.svd(A)
Sigma = np.diag(S)  # 2x2 の対角行列

print("A =")
print(A)
print("\nU =")
print(U)
print("\nS =", S)
print("Sigma =")
print(Sigma)
print("\nVt =")
print(Vt)
print("\nU @ Sigma @ Vt =")
print(U @ Sigma @ Vt)

# 単位円上の点をサンプリング
theta = np.linspace(0, 2*np.pi, 200)
x_circle = np.cos(theta)
y_circle = np.sin(theta)
circle_points = np.vstack([x_circle, y_circle])

# 各ステップでの像を計算
# ステップ1: Vt による回転（入力空間の回転）
rotated_input = Vt @ circle_points

# ステップ2: Sigma による伸縮（軸方向の伸縮）
scaled = Sigma @ rotated_input

# ステップ3: U による回転（出力空間の回転）
final_output = U @ scaled

# 可視化の準備
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
(ax1, ax2), (ax3, ax4) = axes

# ステップ0: 元の単位円
ax1.plot(x_circle, y_circle, 'b-', label='単位円')
ax1.set_title('ステップ0: 入力空間の単位円')
ax1.set_aspect('equal')
ax1.grid(True)
ax1.legend()

# ステップ1: Vt による回転後の単位円
ax2.plot(rotated_input[0], rotated_input[1], 'g-', label='Vt による回転後')
ax2.set_title('ステップ1: 入力空間の回転 (Vt)')
ax2.set_aspect('equal')
ax2.grid(True)
ax2.legend()

# ステップ2: Sigma による伸縮後の楕円
ax3.plot(scaled[0], scaled[1], 'r-', label='Sigma による伸縮後')
ax3.set_title('ステップ2: 軸方向の伸縮 (Sigma)')
ax3.set_aspect('equal')
ax3.grid(True)
ax3.legend()

# ステップ3: U による回転後の最終的な像
ax4.plot(final_output[0], final_output[1], 'm-', label='U による回転後')
ax4.set_title('ステップ3: 出力空間の回転 (U) = A(単位円)')
ax4.set_aspect('equal')
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.show()

# アニメーションで3ステップを連続的に表示（オプション）
fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
ax_anim.set_xlim(-2, 2)
ax_anim.set_ylim(-2, 2)
ax_anim.set_aspect('equal')
ax_anim.grid(True)

lines = []
line0, = ax_anim.plot([], [], 'b-', label='単位円')
line1, = ax_anim.plot([], [], 'g-', label='Vt 回転後')
line2, = ax_anim.plot([], [], 'r-', label='Sigma 伸縮後')
line3, = ax_anim.plot([], [], 'm-', label='U 回転後 = A(単位円)')
lines = [line0, line1, line2, line3]
ax_anim.legend()

def animate(frame):
    # frame=0: 単位円のみ
    # frame=1: 単位円 + Vt回転後
    # frame=2: 単位円 + Vt回転後 + Sigma伸縮後
    # frame=3: 単位円 + Vt回転後 + Sigma伸縮後 + U回転後
    for i, line in enumerate(lines):
        if i <= frame:
            if i == 0:
                line.set_data(x_circle, y_circle)
            elif i == 1:
                line.set_data(rotated_input[0], rotated_input[1])
            elif i == 2:
                line.set_data(scaled[0], scaled[1])
            elif i == 3:
                line.set_data(final_output[0], final_output[1])
        else:
            line.set_data([], [])
    return lines

anim = FuncAnimation(fig_anim, animate, frames=4, interval=1000, blit=True)
plt.show()