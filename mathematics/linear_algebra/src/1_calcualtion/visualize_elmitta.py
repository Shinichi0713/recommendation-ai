import numpy as np
import matplotlib.pyplot as plt

# 1. 適当な複素行列 X を作成
n = 10
X = np.random.randn(n, n) + 1j * np.random.randn(n, n)

# 2. エルミート行列 A を生成 (A = X + X* )
# これにより必ず A = A* が成り立つ
A = X + X.conj().T

# 3. 固有値を計算
eigenvalues = np.linalg.eigvals(A)

# 4. 複素平面上にプロット
plt.figure(figsize=(8, 2))
plt.scatter(eigenvalues.real, eigenvalues.imag, c='red', s=50, label='Eigenvalues')
plt.axhline(0, color='black', lw=1) # 実数軸
plt.axvline(0, color='black', lw=1) # 虚数軸
plt.grid(True, linestyle='--')
plt.title("Eigenvalues of a Hermitian Matrix on the Complex Plane")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.show()
