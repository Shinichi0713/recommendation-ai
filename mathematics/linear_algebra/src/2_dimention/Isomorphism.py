import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_isomorphism():
    # 1. 2つの多項式の係数を定義（これがベクトル空間の成分になる）
    # P1(x) = 1x^2 + 2x + 1
    # P2(x) = -1x^2 + 1x + 2
    v1 = np.array([1, 2, 1])
    v2 = np.array([-1, 1, 2])
    v3 = v1 + v2  # 和のベクトル

    x_range = np.linspace(-2, 2, 100)
    
    def poly(v, x):
        return v[0]*x**2 + v[1]*x + v[2]

    # プロットの作成
    fig = plt.figure(figsize=(14, 6))

    # --- 左側: 多項式空間 V (関数の世界) ---
    ax1 = fig.add_subplot(121)
    ax1.plot(x_range, poly(v1, x_range), 'r-', label='P1(x) = $x^2+2x+1$')
    ax1.plot(x_range, poly(v2, x_range), 'g-', label='P2(x) = $-x^2+x+2$')
    ax1.plot(x_range, poly(v3, x_range), 'b--', lw=3, label='Sum: P1+P2')
    ax1.set_title("Polynomial Space $V$\n(Functions)")
    ax1.grid(True)
    ax1.legend()

    # --- 右側: 数ベクトル空間 R^3 (矢印の世界) ---
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 原点からのベクトルを描画
    origin = np.zeros(3)
    ax2.quiver(*origin, *v1, color='r', label='v1 = [1, 2, 1]')
    ax2.quiver(*origin, *v2, color='g', label='v2 = [-1, 1, 2]')
    ax2.quiver(*origin, *v3, color='b', lw=3, label='v1 + v2')

    # ベクトルの終点を点でも表示
    ax2.scatter(*v1, color='r')
    ax2.scatter(*v2, color='g')
    ax2.scatter(*v3, color='b')

    ax2.set_xlim([-2, 2]); ax2.set_ylim([0, 3]); ax2.set_zlim([0, 4])
    ax2.set_title("Vector Space $\mathbb{R}^3$\n(Coordinates)")
    ax2.set_xlabel('a ($x^2$)'); ax2.set_ylabel('b ($x^1$)'); ax2.set_zlabel('c ($x^0$)')
    ax2.legend()

    plt.suptitle("Isomorphism: $P(x) \longleftrightarrow (a, b, c)$", fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_isomorphism()