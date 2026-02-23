import numpy as np
import matplotlib.pyplot as plt

def simulate_collapsing_space(steps=5):
    # 1. 元の基底ベクトル
    e1 = np.array([1, 0])
    e2_start = np.array([0, 1])  # 最初は直交（行列式=1）
    e2_end = np.array([1, 0])    # 最後はe1と同じ（行列式=0、空間が潰れる）

    # グリッド描画用のデータ
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack([X.flatten(), Y.flatten()])

    fig, axes = plt.subplots(1, steps, figsize=(20, 4))
    
    for i, t in enumerate(np.linspace(0, 1, steps)):
        # e2 を徐々に e1 に近づける
        e2_current = (1 - t) * e2_start + t * e2_end
        
        # 写像行列 A = [e1, e2_current]
        A = np.column_stack([e1, e2_current])
        det = np.linalg.det(A)
        
        # 空間の変形
        t_pts = A @ pts
        TX = t_pts[0, :].reshape(X.shape)
        TY = t_pts[1, :].reshape(Y.shape)
        
        # 描画
        ax = axes[i]
        # 変形後のグリッド
        for j in range(len(x)):
            ax.plot(TX[j, :], TY[j, :], color='gray', alpha=0.3)
            ax.plot(TX[:, j], TY[:, j], color='gray', alpha=0.3)
        
        # ベクトルの描画（複数のベクトルが潰れていく様子）
        ax.quiver(0, 0, A[0,0], A[1,0], color='red', angles='xy', scale_units='xy', scale=1, label='v1')
        ax.quiver(0, 0, A[0,1], A[1,1], color='blue', angles='xy', scale_units='xy', scale=1, label='v2')
        
        # 設定
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f"Det = {det:.2f}")
        if i == 0: ax.legend()

    plt.suptitle("Space collapsing from 2D to 1D as Determinant approaches 0", fontsize=16)
    plt.tight_layout()
    plt.show()

# シミュレーション実行
simulate_collapsing_space(steps=5)