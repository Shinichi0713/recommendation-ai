import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. ユニタリ行列（回転行列）の生成
# 任意の軸周りの回転は直交行列（実ユニタリ行列）になります
def get_rotation_matrix(alpha, beta, gamma):
    # X, Y, Z軸周りの回転行列を合成
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

# 適当な角度で回転行列を作成
U = get_rotation_matrix(np.pi/4, np.pi/6, np.pi/3)

# 2. 単位球面のデータ作成
u_grid = np.linspace(0, 2 * np.pi, 30)
v_grid = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u_grid), np.sin(v_grid))
y = np.outer(np.sin(u_grid), np.sin(v_grid))
z = np.outer(np.ones(np.size(u_grid)), np.cos(v_grid))

# 3. 球面上の点を変換
points = np.stack([x.flatten(), y.flatten(), z.flatten()])
transformed_points = U @ points
tx = transformed_points[0].reshape(x.shape)
ty = transformed_points[1].reshape(y.shape)
tz = transformed_points[2].reshape(z.shape)

# 4. 可視化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 変換後の球面（形が変わっていないことを示す）
ax.plot_surface(tx, ty, tz, color='lightgreen', alpha=0.2, edgecolor='forestgreen', linewidth=0.3)

# 元の座標軸 (X:Red, Y:Green, Z:Blue)
# 変換後の座標軸 (Uの各列ベクトル)
colors = ['r', 'g', 'b']
labels = ['X', 'Y', 'Z']
for i in range(3):
    # 元の基底ベクトル（細い点線）
    base = np.zeros(3)
    vec_orig = np.zeros(3); vec_orig[i] = 1
    ax.quiver(0, 0, 0, vec_orig[0], vec_orig[1], vec_orig[2], 
              color=colors[i], linestyle='--', alpha=0.3, arrow_length_ratio=0.1)
    
    # 変換後の基底ベクトル（太い実線 = Uの列ベクトル）
    vec_trans = U[:, i]
    ax.quiver(0, 0, 0, vec_trans[0], vec_trans[1], vec_trans[2], 
              color=colors[i], lw=3, label=f'Transformed {labels[i]} axis', arrow_length_ratio=0.1)

# グラフの装飾
ax.set_title("Unitary Transformation: Pure Rotation (Length & Angle Preserved)")
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
ax.view_init(elev=20, azim=45)

plt.show()