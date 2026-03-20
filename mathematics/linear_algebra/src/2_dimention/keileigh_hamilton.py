import numpy as np

# 1. 適当な3次正方行列 A を作成
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 2. 固有多項式の係数を求める
# np.poly(A) は、det(xI - A) = x^n + c_{n-1}x^{n-1} + ... + c0 の係数 [1, c_{n-1}, ..., c0] を返す
coeffs = np.poly(A)
print(f"固有多項式の係数: {coeffs}")

# 3. 固有多項式 P(A) = c_n*A^n + ... + c0*E を計算
n = len(A)
PA = np.zeros_like(A, dtype=float)
E = np.eye(n)

# 各項 (coeffs[i] * A^(n-i)) を足し合わせる
for i, c in enumerate(coeffs):
    power = n - i
    if power > 0:
        PA = PA + c * np.linalg.matrix_power(A, power)
    else:
        PA = PA + c * E # 定数項には単位行列を掛ける

# 4. 結果を表示
print("\n計算結果 P(A):")
print(np.round(PA, 10))  # 浮動小数点の誤差を丸めて表示

# 零行列かどうか判定
if np.allclose(PA, 0):
    print("\n結論: P(A) は零行列になりました！定理は成立しています。")