import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Linear_model, Ridge
from sklearn.preprocessing import StandardScaler

# 1. データの作成
np.random.seed(42)
n_samples = 50
n_features = 10

# 入力データ X (50サンプル、10特徴量)
X = np.random.randn(n_samples, n_features)

# 真の重み：10個中、最初の2個だけが重要（3.0と-2.0）、残りはすべて0
true_w = np.array([3.0, -2.0, 0, 0, 0, 0, 0, 0, 0, 0])

# 目的変数 y (ノイズを少し加える)
y = X @ true_w + 0.1 * np.random.randn(n_samples)

# 2. Lasso回帰の実行 (alphaが正則化の強さ λ)
lasso = Lasso(alpha=0.5)
lasso.fit(X, y)

# 比較用に普通の最小二乗法（正則化なし）も実行
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(X, y)

# 3. 結果の可視化
plt.figure(figsize=(10, 6))
plt.stem(np.arange(n_features), true_w, linefmt='g-', markerfmt='go', label='True weights (Goal)')
plt.stem(np.arange(n_features) + 0.2, ols.coef_, linefmt='r-', markerfmt='rx', label='Ordinary Least Squares (No Reg)')
plt.stem(np.arange(n_features) - 0.2, lasso.coef_, linefmt='b-', markerfmt='bs', label='Lasso weights (Alpha=0.5)')

plt.title("Comparison of Weights: OLS vs Lasso")
plt.xlabel("Feature Index")
plt.ylabel("Weight Value")
plt.xticks(np.arange(n_features))
plt.legend()
plt.grid(True)
plt.show()

# 実際に0になった数を確認
print(f"Lassoで0になった変数の数: {np.sum(lasso.coef_ == 0)} / {n_features}")