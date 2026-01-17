import numpy as np
import matplotlib.pyplot as plt

# 1. 凸関数の定義 (f(x) = x^2)
def f(x):
    return x**2

# 2. データの準備
x = np.linspace(-2, 2, 100)
y = f(x)

# 3. 2点 x1, x2 を選ぶ
x1, x2 = -1.5, 1.0
y1, y2 = f(x1), f(x2)

# 4. 内分点 (lambda = 0.4) での比較
lmbda = 0.4
x_internal = lmbda * x1 + (1 - lmbda) * x2
y_curve = f(x_internal)             # 左辺: 曲線の高さ
y_line = lmbda * y1 + (1 - lmbda) * y2  # 右辺: 線分上の高さ

# 5. プロット
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='$f(x) = x^2$ (Curve)', color='blue', linewidth=2)
plt.plot([x1, x2], [y1, y2], 'ro--', label='Secant line (Line segment)')

# 垂直線の描画 (linestyles を 'dotted' に修正)
plt.vlines(x_internal, y_curve, y_line, colors='green', linestyles='dotted', label='Gap (Convexity)')

# 内分点でのポイントを強調
plt.scatter([x_internal], [y_curve], color='blue', zorder=5) # 曲線上の点
plt.scatter([x_internal], [y_line], color='red', zorder=5)  # 線分上の点

# ラベルと注釈の設定
plt.title("Visualizing Convex Function Definition")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.annotate('Curve: $f(\lambda x + (1-\lambda)y)$', xy=(x_internal, y_curve), 
             xytext=(x_internal+0.2, y_curve-0.5), arrowprops=dict(arrowstyle='->'))
plt.annotate('Line: $\lambda f(x) + (1-\lambda)f(y)$', xy=(x_internal, y_line), 
             xytext=(x_internal+0.2, y_line+0.5), arrowprops=dict(arrowstyle='->'))

plt.legend()
plt.grid(True)
plt.show()