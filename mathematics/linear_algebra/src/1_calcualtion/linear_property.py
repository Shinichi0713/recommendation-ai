import numpy as np
import matplotlib.pyplot as plt

def visualize_linearity(A):
    # --- 1. グリッドデータの作成 ---
    # -2から2までの範囲に11本の線を引く
    range_val = 2
    ticks = np.linspace(-range_val, range_val, 11)
    
    # グリッドの各線上の点を生成する関数
    def get_grid_lines():
        lines = []
        # 垂直線と水平線
        for t in ticks:
            lines.append(np.array([[t, t], [-range_val, range_val]])) # 垂直
            lines.append(np.array([[-range_val, range_val], [t, t]])) # 水平
        # 「まっすぐさ」を強調するための斜め線
        lines.append(np.array([[-range_val, range_val], [-range_val, range_val]]))
        return lines

    original_lines = get_grid_lines()

    # --- 2. 可視化の準備 ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    titles = ['Original Space (Standard Grid)', 'Transformed Space (Linear Map)']
    colors = ['#1f77b4', '#ff7f0e'] # 青とオレンジ

    for i, title in enumerate(titles):
        ax[i].set_title(title, fontsize=14)
        ax[i].set_xlim(-5, 5)
        ax[i].set_ylim(-5, 5)
        ax[i].axhline(0, color='black', lw=1.5, alpha=0.5) # X軸
        ax[i].axvline(0, color='black', lw=1.5, alpha=0.5) # Y軸
        ax[i].set_aspect('equal')
        ax[i].grid(True, linestyle=':', alpha=0.3)

    # --- 3. 変換と描画 ---
    for line in original_lines:
        # 元の線を描画
        ax[0].plot(line[0], line[1], color='gray', lw=1, alpha=0.6)
        
        # 行列 A による線形写像を適用
        # lineは [[x1, x2], [y1, y2]] なので、各列ベクトルにAを掛ける
        transformed_line = A @ line
        
        # 変換後の線を描画
        # 線形写像なら、この transformed_line も「直線」になる
        ax[1].plot(transformed_line[0], transformed_line[1], color='orange', lw=1.5)

    # 原点の強調
    ax[0].plot(0, 0, 'ko', markersize=8, label='Origin')
    ax[1].plot(0, 0, 'ko', markersize=8, label='Origin')
    
    plt.tight_layout()
    plt.show()

# --- 4. 線形写像を定義して実行 ---
# 例: 剪断(Shear)とスケーリングを組み合わせた行列
# 1列目が基底e1の行き先、2列目が基底e2の行き先
A = np.array([
    [1.5, 1.0], 
    [0.5, 1.2]
])

visualize_linearity(A)