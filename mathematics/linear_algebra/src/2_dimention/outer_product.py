import numpy as np
import matplotlib.pyplot as plt

# 1. 2つのベクトルの定義
a = np.array([3, 0, 0])
b = np.array([0, 4, 0])

# 2. 外積の計算
cross_prod = np.cross(a, b)
area = np.linalg.norm(cross_prod)

# 3. 3D可視化の設定
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
o = np.zeros(3)

# 元のベクトル a, b の描画
ax.quiver(*o, *a, color='blue', label=f'Vector a {a}', lw=3)
ax.quiver(*o, *b, color='red', label=f'Vector b {b}', lw=3)

# 外積ベクトル a x b の描画 (面に垂直)
ax.quiver(*o, *cross_prod, color='green', label=f'Cross Product (Area={area:.1f})', lw=3)

# 平行四辺形の面を塗りつぶし (a, b が作る面)
# 四角形の頂点: 原点, a, a+b, b
vertices = np.array([o, a, a + b, b])
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
poly = Poly3DCollection([vertices], alpha=0.3, facecolor='yellow', edgecolor='orange')
ax.add_collection3d(poly)

# 4. グラフの調整
ax.set_xlim([-1, 5]); ax.set_ylim([-1, 5]); ax.set_zlim([-1, 13])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.title(f"Cross Product Visualization\nDirection: Perpendicular | Magnitude: Area of Parallelogram")
ax.view_init(elev=20, azim=45)
plt.legend()
plt.show()

print(f"ベクトル a: {a}")
print(f"ベクトル b: {b}")
print(f"外積 a x b: {cross_prod}")
print(f"平行四辺形の面積 (外積の大きさ): {area}")