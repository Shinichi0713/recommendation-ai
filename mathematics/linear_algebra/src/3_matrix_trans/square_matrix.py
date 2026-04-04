import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2次形式 Q(x) = x^T A x を定義
def quadratic_form(x, A):
    return x.T @ A @ x

# 等高線と3D曲面を描画する関数
def plot_quadratic_form(A, title_suffix=""):
    # グリッド生成
    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-2, 2, 50)
    X1, X2 = np.meshgrid(x1, x2)
    
    # 2次形式の値を計算
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i,j], X2[i,j]])
            Z[i,j] = quadratic_form(x, A)
    
    # 固有値と符号を表示
    eigvals = np.linalg.eigvalsh(A)
    sign_type = "正定値" if np.all(eigvals > 0) else \
               "負定値" if np.all(eigvals < 0) else \
               "不定"
    
    # 等高線プロット
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    contour = plt.contour(X1, X2, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'2次形式の等高線 ({sign_type}) {title_suffix}')
    plt.grid(True)
    plt.axis('equal')
    
    # 3D曲面プロット
    ax = plt.subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$Q(x)$')
    ax.set_title(f'2次形式の3D曲面 ({sign_type}) {title_suffix}')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"行列 A = {A}")
    print(f"固有値: {eigvals}")
    print(f"符号: {sign_type}\n")

# 例1: 正定値 (楕円的な等高線)
A1 = np.array([[2, 1],
               [1, 2]])
plot_quadratic_form(A1, "例1: 正定値")

# 例2: 負定値 (下に凸な放物面)
A2 = np.array([[-1, 0],
               [0, -2]])
plot_quadratic_form(A2, "例2: 負定値")

# 例3: 不定 (鞍点)
A3 = np.array([[1, 2],
               [2, -1]])
plot_quadratic_form(A3, "例3: 不定")

# 例4: 半正定値 (放物線的)
A4 = np.array([[1, 1],
               [1, 1]])
plot_quadratic_form(A4, "例4: 半正定値")