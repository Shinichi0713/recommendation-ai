import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14

# 凸関数の例: f(x) = x^2
def f(x):
    return x**2

# 導関数（勾配）: f'(x) = 2x
def grad_f(x):
    return 2 * x

# 接線（一次近似）を返す関数
def tangent_line(x0, x):
    return f(x0) + grad_f(x0) * (x - x0)

# 可視化する範囲
x_plot = np.linspace(-2, 2, 300)
y_true = f(x_plot)

# 接点
x0 = 0.5
y0 = f(x0)

# 接線
y_tangent = tangent_line(x0, x_plot)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(x_plot, y_true, "b-", label="$f(x) = x^2$", linewidth=2)
ax.plot(x_plot, y_tangent, "r--", label=f"接線 at $x={x0}$", linewidth=2)
ax.scatter([x0], [y0], color="red", s=50, zorder=5)

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("凸関数のグラフは接線より上にある（一次近似が下から支えられる）")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()