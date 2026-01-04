import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. データの作成
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)

# 方向部分空間 W (例: z = x + y, 原点を通る)
W_Z = X + Y

# アフィン部分空間 A (例: v = [0, 0, 10] だけ平行移動)
v_z = 10
A_Z = W_Z + v_z

# 2. 描画設定
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 方向部分空間 W のプロット（青色・半透明）
ax.plot_surface(X, Y, W_Z, color='blue', alpha=0.3, label='Directional Subspace (W)')

# アフィン部分空間 A のプロット（赤色・半透明）
ax.plot_surface(X, Y, A_Z, color='red', alpha=0.5, label='Affine Subspace (A = v + W)')

# 原点と平行移動ベクトルの描画
ax.quiver(0, 0, 0, 0, 0, v_z, color='black', lw=2, arrow_length_ratio=0.1, label='Translation Vector (v)')
ax.scatter(0, 0, 0, color='black', s=50) # 原点

# グラフの装飾
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualization of Affine Subspace')
# 凡例はplot_surfaceで直接出せないので、ダミーを作成
import matplotlib.lines as mlines
blue_proxy = mlines.Line2D([], [], color='blue', alpha=0.3, label='Directional Subspace (W)')
red_proxy = mlines.Line2D([], [], color='red', alpha=0.5, label='Affine Subspace (A)')
ax.legend(handles=[blue_proxy, red_proxy])

plt.show()