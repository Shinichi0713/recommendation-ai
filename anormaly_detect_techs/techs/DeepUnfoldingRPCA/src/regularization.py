import numpy as np
import matplotlib.pyplot as plt

def soft_threshold(x, lam):
    """L1正則化で使われる近接演算（ソフトしきい値）"""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def l2_update(x, grad, lam, lr):
    """L2正則化（リッジ）の更新式"""
    # 勾配に 2*lam*x が加わる
    return x - lr * (grad + 2 * lam * x)

# パラメータ設定
initial_x = 1.5   # 初期値
target_x = 0.0    # データの勾配が 0 を指す地点（本来ここへ行きたい）
lr = 0.1          # 学習率
lam = 0.2         # 正則化の強さ
iterations = 50

# 履歴の保存
x_l1_hist = [initial_x]
x_l2_hist = [initial_x]

curr_x_l1 = initial_x
curr_x_l2 = initial_x

for i in range(iterations):
    # シンプルな2乗誤差の勾配 (x - target_x)
    grad_l1 = curr_x_l1 - target_x
    grad_l2 = curr_x_l2 - target_x
    
    # L1更新: 勾配降下した後にソフトしきい値を適用 (ISTAの基本形)
    curr_x_l1 = soft_threshold(curr_x_l1 - lr * grad_l1, lr * lam)
    
    # L2更新: 通常の勾配降下
    curr_x_l2 = l2_update(curr_x_l2, grad_l2, lam, lr)
    
    x_l1_hist.append(curr_x_l1)
    x_l2_hist.append(curr_x_l2)

# --- 可視化 ---
plt.figure(figsize=(12, 6))

# 収束プロット
plt.plot(x_l1_hist, 'o-', label=f'L1 (Lasso) - $\lambda$={lam}', color='red', markersize=4)
plt.plot(x_l2_hist, 's-', label=f'L2 (Ridge) - $\lambda$={lam}', color='blue', markersize=4)

plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.title("Convergence Comparison: L1 vs L2 Regularization")
plt.xlabel("Iterations")
plt.ylabel("Value of x")
plt.grid(True, alpha=0.3)
plt.legend()

# 拡大図（0付近の挙動）
plt.axes([0.55, 0.25, 0.3, 0.3])
plt.plot(x_l1_hist[-20:], 'o-', color='red', markersize=4)
plt.plot(x_l2_hist[-20:], 's-', color='blue', markersize=4)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.title("Focus: Near Zero")

plt.show()