import numpy as np
import matplotlib.pyplot as plt

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


alpha = 1.0

I = np.eye(d)
w_closed = np.linalg.inv(X.T @ X + n * alpha * I) @ X.T @ y

def ridge_gradient_descent(X, y, alpha, lr=0.1, max_iter=100):
    n, d = X.shape
    w = np.zeros(d)

    w_history = []
    obj_history = []

    for _ in range(max_iter):
        # 勾配
        grad = -(X.T @ (y - X @ w)) / n + alpha * w

        # 更新
        w = w - lr * grad

        # 記録
        w_history.append(w.copy())

        loss = (1/(2*n)) * np.sum((y - X @ w)**2)
        reg = (alpha/2) * np.sum(w**2)
        obj_history.append(loss + reg)

    return np.array(w_history), np.array(obj_history)

w_hist, obj_hist = ridge_gradient_descent(X, y, alpha, lr=0.2, max_iter=50)

plt.figure()
plt.plot(obj_hist)
plt.xlabel("Iteration")
plt.ylabel("Objective value")
plt.title("Ridge: Objective Convergence")
plt.show()

plt.figure()
plt.plot(w_hist[:, 0], label="w0")
plt.plot(w_hist[:, 1], label="w1")
plt.axhline(w_closed[0], linestyle="--")
plt.axhline(w_closed[1], linestyle="--")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Coefficient value")
plt.title("Coefficient Convergence to Closed-form Solution")
plt.show()

def ridge_objective(w, X, y, alpha):
    n = X.shape[0]
    loss = (1/(2*n)) * np.sum((y - X @ w)**2)
    reg = (alpha/2) * np.sum(w**2)
    return loss + reg

w0 = np.linspace(-1, 5, 100)
w1 = np.linspace(-5, 1, 100)

W0, W1 = np.meshgrid(w0, w1)
Z = np.zeros_like(W0)

for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        w = np.array([W0[i, j], W1[i, j]])
        Z[i, j] = ridge_objective(w, X, y, alpha)

plt.figure(figsize=(7, 6))

# 等高線
plt.contour(W0, W1, Z, levels=30)

# 最適化軌跡
plt.plot(w_hist[:, 0], w_hist[:, 1], 'o-', label="GD path")

# 真の解
plt.plot(w_closed[0], w_closed[1], 'r*', markersize=12, label="Closed-form solution")

plt.xlabel("w0")
plt.ylabel("w1")
plt.title("Optimization Path on Ridge Loss Surface")
plt.legend()
plt.show()
