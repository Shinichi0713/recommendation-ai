import numpy as np
import matplotlib.pyplot as plt

def visualize_eigenvalues():
    # 1. 可視化する行列 A の定義
    # ここでは固有値が 3 と 1、固有ベクトルが [1, 1] と [1, -1] 方向になる行列を使用
    A = np.array([[2, 1],
                  [1, 2]])

    # 固有値と固有ベクトルを計算
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # 2. 単位円上の点を生成（入力ベクトル群）
    theta = np.linspace(0, 2*np.pi, 100)
    circle_points = np.array([np.cos(theta), np.sin(theta)])

    # 3. 行列 A による変形後の点を計算
    transformed_points = A @ circle_points

    # 描画設定
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 元の円と変形後の楕円を描画
    ax.plot(circle_points[0], circle_points[1], color='gray', linestyle='--', label='Original (Unit Circle)')
    ax.plot(transformed_points[0], transformed_points[1], color='blue', label='Transformed (by A)', linewidth=2)

    # 固有ベクトルの描画
    colors = ['red', 'green']
    for i in range(len(eigenvalues)):
        val = eigenvalues[i]
        vec = eigenvectors[:, i]
        
        # 固有ベクトル方向の線（無限に続く軸）
        ax.axline((0, 0), (vec[0], vec[1]), color=colors[i], alpha=0.3, linestyle=':')
        
        # 実際の固有ベクトル（元の長さ）
        ax.quiver(0, 0, vec[0], vec[1], color=colors[i], angles='xy', scale_units='xy', scale=1, 
                  label=f'Eigenvector {i+1} (λ={val:.1f})')
        
        # 変形後の固有ベクトル（固有値倍された長さ）
        transformed_vec = val * vec
        ax.quiver(0, 0, transformed_vec[0], transformed_vec[1], color=colors[i], 
                  angles='xy', scale_units='xy', scale=1, alpha=0.5)

    # グラフのレイアウト調整
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.axhline(0, color='black', lw=1); ax.axvline(0, color='black', lw=1)
    ax.set_title(f'Linear Transformation: Eigenvalue Visualization\nMatrix A = {A.tolist()}', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.show()

if __name__ == "__main__":
    visualize_eigenvalues()