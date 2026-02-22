import numpy as np
import matplotlib.pyplot as plt

def visualize_basis(A):
    # 1. 標準基底 (Standard Basis)
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    # 2. 写像後の基底 (Transformed Basis)
    # 行列 A の各列が、基底の「行き先」になる
    v1 = A @ e1
    v2 = A @ e2
    
    # 3. グリッド線の生成
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    
    # グリッドの全点を一括で変換
    pts = np.vstack([X.flatten(), Y.flatten()])
    t_pts = A @ pts
    TX = t_pts[0, :].reshape(X.shape)
    TY = t_pts[1, :].reshape(Y.shape)

    # 4. プロット
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, (grid_x, grid_y, title, b1, b2) in enumerate([
        (X, Y, "Original Space (Standard Basis)", e1, e2),
        (TX, TY, "Transformed Space (New Basis)", v1, v2)
    ]):
        # グリッド線の描画
        for j in range(len(x)):
            ax[i].plot(grid_x[j, :], grid_y[j, :], color='gray', alpha=0.3, lw=1)
            ax[i].plot(grid_x[:, j], grid_y[:, j], color='gray', alpha=0.3, lw=1)
        
        # 基底ベクトルの描画
        ax[i].quiver(0, 0, b1[0], b1[1], color='red', angles='xy', scale_units='xy', scale=1, label=f'Basis 1: {b1}')
        ax[i].quiver(0, 0, b2[0], b2[1], color='blue', angles='xy', scale_units='xy', scale=1, label=f'Basis 2: {b2}')
        
        ax[i].set_xlim(-4, 4); ax[i].set_ylim(-4, 4)
        ax[i].axhline(0, color='black', lw=1); ax[i].axvline(0, color='black', lw=1)
        ax[i].set_title(title)
        ax[i].legend()
        ax[i].grid(True, linestyle=':', alpha=0.6)
        ax[i].set_aspect('equal')

    plt.tight_layout()
    plt.show()

# --- 実行 ---
# 例: せん断(Shear)と拡大を組み合わせた行列
# 1列目が「赤色矢印の行き先」、2列目が「青色矢印の行き先」
A = np.array([[2, 1], 
              [0, 1.5]])

visualize_basis(A)