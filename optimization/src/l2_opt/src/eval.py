import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


np.random.seed(42)

n_samples = 200
n_features = 50
n_informative = 5

# 入力データ
X = np.random.randn(n_samples, n_features)

# 真の係数（スパース）
true_coef = np.zeros(n_features)
true_coef[:n_informative] = [5, -4, 3, 2, -1]

# 出力（ノイズ付き）
y = X @ true_coef + 0.5 * np.random.randn(n_samples)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)

lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)

lasso_coef = lasso.coef_
ridge_coef = ridge.coef_

plt.figure(figsize=(12, 6))

plt.plot(lasso_coef, 'o', label='Lasso', alpha=0.8)
plt.plot(ridge_coef, 'x', label='Ridge', alpha=0.8)
plt.axhline(0)

plt.title("Lasso vs Ridge: Coefficient Comparison")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.show()

lasso_zero = np.sum(lasso_coef == 0)
ridge_zero = np.sum(np.isclose(ridge_coef, 0, atol=1e-3))

print("Lasso zero coefficients:", lasso_zero)
print("Ridge near-zero coefficients:", ridge_zero)

alphas = np.logspace(-2, 1, 50)

lasso_path = []
ridge_path = []

for a in alphas:
    l = Lasso(alpha=a, max_iter=10000)
    r = Ridge(alpha=a)
    
    l.fit(X_train, y_train)
    r.fit(X_train, y_train)
    
    lasso_path.append(l.coef_)
    ridge_path.append(r.coef_)

lasso_path = np.array(lasso_path)
ridge_path = np.array(ridge_path)

plt.figure(figsize=(12, 6))

for i in range(n_features):
    plt.plot(alphas, lasso_path[:, i], color='blue', alpha=0.2)

plt.xscale('log')
plt.title("Lasso Coefficient Paths (many become exactly zero)")
plt.xlabel("alpha")
plt.ylabel("Coefficient")
plt.show()

plt.figure(figsize=(12, 6))

for i in range(n_features):
    plt.plot(alphas, ridge_path[:, i], color='red', alpha=0.2)

plt.xscale('log')
plt.title("Ridge Coefficient Paths (shrink but rarely zero)")
plt.xlabel("alpha")
plt.ylabel("Coefficient")
plt.show()
