import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D # 3Dプロットに必要

# 1. データの生成
np.random.seed(42)
n_samples = 100
base_data = np.random.randn(n_samples, 2)

x = base_data[:, 0]
y = base_data[:, 1]
# z = 0.5x + 0.8y + ノイズ（これが「ほぼ平面」を作る）
z = 0.5 * x + 0.8 * y + np.random.normal(0, 0.1, n_samples) 

data_3d = np.vstack([x, y, z]).T

# 2. PCAで「本質的な部分空間（2次元平面）」を見つける
pca = PCA(n_components=2)
pca.fit(data_3d)
basis_vectors = pca.components_

# 3. 可視化
fig = plt.figure(figsize=(10, 8))
# 修正箇所: add_subplot を使用
ax = fig.add_subplot(111, projection='3d')

# 元のデータ点をプロット
ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], alpha=0.6, label='Original Data')

# 見つけ出した「部分空間（平面）」を描画
grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
normal = np.cross(basis_vectors[0], basis_vectors[1])
grid_z = -(normal[0] * grid_x + normal[1] * grid_y) / normal[2]
ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.2, color='orange')

# 部分空間を支える「基底ベクトル」を矢印で表示
for i, v in enumerate(basis_vectors):
    ax.quiver(0, 0, 0, v[0]*2, v[1]*2, v[2]*2, color='red', lw=3, label=f'Basis Vector {i+1}')

ax.set_title("Identifying a 2D Subspace in 3D Data")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.show()