import numpy as np
import matplotlib.pyplot as plt

def visualize_function_space(a, b, c):
    # x軸の範囲（定義域）
    x = np.linspace(-2, 2, 100)
    
    # 抽象ベクトル空間の「基底」となる関数たち
    phi0 = np.ones_like(x)   # 基底1: 1 (定数関数)
    phi1 = x                # 基底2: x (1次関数)
    phi2 = x**2             # 基底3: x^2 (2次関数)
    
    # 基底のスカラー倍（成分との掛け算）
    v0 = a * phi0
    v1 = b * phi1
    v2 = c * phi2
    
    # ベクトルの和（関数の足し算）
    f_x = v0 + v1 + v2
    
    # 描画
    plt.figure(figsize=(10, 6))
    
    # 各成分（スカラー倍された基底）の描画
    plt.plot(x, v0, '--', label=f'Component 1: {a} * (1)', alpha=0.5)
    plt.plot(x, v1, '--', label=f'Component 2: {b} * (x)', alpha=0.5)
    plt.plot(x, v2, '--', label=f'Component 3: {c} * (x^2)', alpha=0.5)
    
    # 合成された「ベクトル」としての関数
    plt.plot(x, f_x, color='red', lw=3, label=f'Resultant Vector f(x) = {a} + {b}x + {c}x^2')
    
    # グラフの装飾
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.title("Visualization of a Function as an Abstract Vector")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

# --- 実行 ---
# 係数 (a, b, c) を変えることで、空間内の異なるベクトルを生成できます
# 例: f(x) = 1 + 0.5x + 2x^2
visualize_function_space(a=1, b=0.5, c=2)