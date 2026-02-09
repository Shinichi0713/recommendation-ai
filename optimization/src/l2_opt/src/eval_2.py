
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


np.random.seed(0)

# サンプル数と特徴数（2次元で可視化）
n = 100
d = 2

X = np.random.randn(n, d)
true_w = np.array([3.0, -2.0])

y = X @ true_w + 0.5 * np.random.randn(n)

# 標準化
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = y - y.mean()


print(X, y)

