import matplotlib.pyplot as plt
import numpy as np

# 1. データの準備
x = np.linspace(0, 20, 400)

# 制約条件の式を y = ... の形に変形
# リンゴ: x + 2y <= 20  => y <= (20 - x) / 2
y1 = (20 - x) / 2
# オレンジ: 2x + y <= 20  => y <= 20 - 2*x
y2 = 20 - 2*x

# 2. グラフの描画設定
plt.figure(figsize=(8, 8))
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.xlabel('Special Mix (x)')
plt.ylabel('Rich Blend (y)')
plt.title('Production Optimization (Linear Programming)')

# 3. 制約線のプロット
plt.plot(x, y1, label='Apple Constraint (x + 2y <= 20)', color='blue', lw=2)
plt.plot(x, y2, label='Orange Constraint (2x + y <= 20)', color='red', lw=2)

# 4. 実行可能領域（Feasible Region）の塗りつぶし
# y1, y2, および 0 のうち、最も小さい値をとる範囲を塗る
y3 = np.minimum(y1, y2)
plt.fill_between(x, 0, y3, where=(y3>=0), color='gray', alpha=0.3, label='Feasible Region')

# 5. 最適解 (20/3, 20/3) のプロット
opt_x, opt_y = 20/3, 20/3
plt.plot(opt_x, opt_y, 'go', markersize=10, label=f'Optimal Point ({opt_x:.2f}, {opt_y:.2f})')

# 6. 仕上げ
plt.annotate(f'Max Profit: 6,000 yen\nx={opt_x:.2f}, y={opt_y:.2f}', 
             xy=(opt_x, opt_y), xytext=(opt_x+1, opt_y+1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()