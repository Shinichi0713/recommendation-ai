import numpy as np
import matplotlib.pyplot as plt

def visualize_rotation(degree):
    # 1. 角度をラジアンに変換
    theta = np.radians(degree)
    
    # 2. 回転行列 R の定義
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    # 3. グリッド線の生成
    x = np.linspace(-2, 2, 11)
    y = np.linspace(-2, 2, 11)
    X, Y = np.meshgrid(x, y)
    
    # グリッドの全点を回転行列で変換
    pts = np.vstack([X.flatten(), Y.flatten()])
    t_pts = R @ pts
    TX = t_pts[0, :].reshape(X.shape)
    TY = t_pts[1, :].reshape(Y.shape)

    # 4. プロット
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 変換後のグリッド線（回転した格子）
    for i in range(len(x)):
        ax.plot(TX[i, :], TY[i, :], color='gray', alpha=0.3, lw=1)
        ax.plot(TX[:, i], TY[:, i], color='gray', alpha=0.3, lw=1)
    
    # 標準基底の「成れの果て」を描画
    e1_transformed = R @ np.array([1, 0])
    e2_transformed = R @ np.array([0, 1])
    
    # 軌跡（単位円）を描画
    circle = plt.Circle((0, 0), 1, color='lightgray', fill=False, linestyle='--')
    ax.add_artist(circle)

    # ベクトルの描画
    ax.quiver(0, 0, e1_transformed[0], e1_transformed[1], color='red', 
              angles='xy', scale_units='xy', scale=1, label=f'R({degree}°) e1')
    ax.quiver(0, 0, e2_transformed[0], e2_transformed[1], color='blue', 
              angles='xy', scale_units='xy', scale=1, label=f'R({degree}°) e2')
    
    # グラフの装飾
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    ax.set_aspect('equal')
    ax.set_title(f"Rotation Matrix Visualization (theta = {degree}°)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.show()

# --- 実行 ---
# 好きな角度を入れてみてください（例：45度）
visualize_rotation(45)