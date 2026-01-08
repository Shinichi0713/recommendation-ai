import numpy as np
import matplotlib.pyplot as plt

def soft_thresholding(x, theta):
    """ソフトしきい値関数: 小さい値を0に、大きい値を0方向に引き寄せる"""
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0)

def ista(y, D, alpha, iterations, theta):
    """
    ISTAアルゴリズム
    y: 観測信号 (input_dim,)
    D: 辞書行列 (input_dim, hidden_dim)
    alpha: ステップサイズ (1 / L, LはD^T Dの最大固有値)
    iterations: 繰り返しの回数
    theta: しきい値 (lambda * alpha)
    """
    x = np.zeros(D.shape[1]) # 推定するスパース信号の初期値
    
    loss_history = []
    
    for i in range(iterations):
        # 1. 勾配計算（データを再現しようとする修正指示）
        # x_next = x + alpha * D.T * (y - D*x)
        residual = y - np.dot(D, x)
        gradient_step = x + alpha * np.dot(D.T, residual)
        
        # 2. ソフトしきい値処理（スパース化：値を0に削る）
        x = soft_thresholding(gradient_step, theta)
        
        # 記録用：再構成誤差の計算
        loss = 0.5 * np.sum((y - np.dot(D, x))**2) + (theta/alpha) * np.sum(np.abs(x))
        loss_history.append(loss)
        
    return x, loss_history

# --- テストデータの準備 ---
np.random.seed(42)
input_dim = 50   # 観測データの次元
hidden_dim = 100 # 辞書の要素数（パーツの数）

# 1. 辞書 D の作成（ランダム）
D = np.random.randn(input_dim, hidden_dim)
# ステップサイズ alpha の計算（収束のために D^T D の最大固有値の逆数より小さくする）
L = np.linalg.norm(D, ord=2)**2
alpha = 1.0 / L

# 2. 真のスパース信号 x_true の作成（10個だけ値があるトゲトゲ信号）
x_true = np.zeros(hidden_dim)
indices = np.random.choice(hidden_dim, 10, replace=False)
x_true[indices] = np.random.randn(10)

# 3. 観測信号 y の作成 (y = Dx + ノイズ)
y = np.dot(D, x_true) + np.random.randn(input_dim) * 0.05

# --- ISTA の実行 ---
iterations = 200
theta = 0.01 # スパース性の強さ
x_recovered, history = ista(y, D, alpha, iterations, theta)

# --- 結果の可視化 ---
plt.figure(figsize=(15, 5))

# 信号の復元具合
plt.subplot(1, 2, 1)
plt.stem(x_true, linefmt='g-', markerfmt='go', label='True (Answer)')
plt.stem(x_recovered, linefmt='r--', markerfmt='rx', label='Recovered (ISTA)')
plt.title(f"Signal Recovery (Iterations: {iterations})")
plt.legend()

# 誤差の収束
plt.subplot(1, 2, 2)
plt.plot(history)
plt.yscale('log')
plt.title("Optimization Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()