import numpy as np
import matplotlib.pyplot as plt

def plot_transform(matrix, title, ax):
    # 格子点（ベクトルのかたまり）を作成
    x, y = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    pts = np.vstack([x.flatten(), y.flatten()])
    
    # 行列による変換（写像）を実行： Y = A @ X
    transformed_pts = matrix @ pts
    
    # 描画
    ax.scatter(pts[0], pts[1], c='gray', alpha=0.3, s=10, label='Original')
    ax.scatter(transformed_pts[0], transformed_pts[1], c='blue', s=15, label='Transformed')
    
    # 基底ベクトルの行き先を矢印で表示
    v_x = matrix @ np.array([1, 0])
    v_y = matrix @ np.array([0, 1])
    ax.quiver(0, 0, v_x[0], v_x[1], color='red', angles='xy', scale_units='xy', scale=1, label='Basis X')
    ax.quiver(0, 0, v_y[0], v_y[1], color='green', angles='xy', scale_units='xy', scale=1, label='Basis Y')
    
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.axhline(0, color='black', lw=1); ax.axvline(0, color='black', lw=1)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# 1. 正則な行列（回転と拡大）
# 行列式 det(A) = 1*1 - 0.5*0 = 1 (0ではない)
A_regular = np.array([[1, 0.5], 
                      [0, 1]])

# 2. 正則ではない行列（一方の次元へ押しつぶす）
# 行列式 det(A) = 1*1 - 1*1 = 0 (空間が潰れる)
A_singular = np.array([[1, 2], 
                       [0.5, 1]])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_transform(A_regular, "Regular Matrix (Invertible)\nSpace is preserved", ax1)
plot_transform(A_singular, "Singular Matrix (Non-invertible)\nSpace collapses to a line", ax2)
plt.show()