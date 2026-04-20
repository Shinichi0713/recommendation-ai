import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14
np.random.seed(123)

# 問題サイズ
n = 20
m = 50

# ランダムな行列 A と真の解 x_true
A = np.random.randn(m, n)
x_true = np.random.randn(n)
b = A @ x_true

# 目的関数 f(x) = 0.5 * ||Ax - b||^2
def f(x):
    return 0.5 * np.linalg.norm(A @ x - b)**2

# 勾配（滑らかなので劣勾配も同じ）
def grad_f(x):
    return A.T @ (A @ x - b)

max_iter = 200
x0 = np.zeros(n)  # 初期点

# ステップサイズ（固定）
L = np.linalg.norm(A.T @ A, 2)  # リプシッツ定数の上界
t = 0.5 / L  # 安定するステップサイズ

# 履歴を保存するリスト
f_history_gd = []   # 勾配法
f_history_sub = []  # 劣勾配法

x_gd = x0.copy()
x_sub = x0.copy()

for k in range(max_iter):
    # 勾配法の更新
    x_gd = x_gd - t * grad_f(x_gd)
    f_history_gd.append(f(x_gd))
    
    # 劣勾配法の更新（滑らかなので g_k = grad_f(x_sub)）
    g_sub = grad_f(x_sub)  # 劣勾配集合は {grad_f(x_sub)} のみ
    x_sub = x_sub - t * g_sub
    f_history_sub.append(f(x_sub))

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.semilogy(f_history_gd, label="Gradient Descent", linewidth=2)
ax.semilogy(f_history_sub, label="Subgradient Method", linestyle="--", linewidth=2)

ax.set_xlabel("Iteration")
ax.set_ylabel("Objective value $f(x_k)$")
ax.set_title("Convergence: Gradient Descent vs Subgradient Method")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()