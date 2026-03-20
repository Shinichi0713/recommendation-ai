import numpy as np

# 1. 推移行列 A の定義
# 都市(U)から：0.9が残留, 0.1が地方へ
# 地方(V)から：0.2が都市へ, 0.8が残留
A = np.array([[0.9, 0.2],
              [0.1, 0.8]])

# 2. 初期人口（都市 1000人、地方 0人）
population = np.array([1000, 0])

# --- パターンA: 地道にシミュレーション ---
current = population.copy()
for i in range(100):
    current = A @ current
print(f"100年後の人口（逐次計算）: 都市 {current[0]:.1f}人, 地方 {current[1]:.1f}人")

# --- パターンB: 固有値・固有ベクトルで一発計算 ---
# 安定状態とは A * x = 1 * x となる状態、つまり「固有値 1」に対応する固有ベクトル
eigenvalues, eigenvectors = np.linalg.eig(A)

# 固有値 1.0 に対応するインデックスを探す
idx = np.argmin(np.abs(eigenvalues - 1.0))
stable_vector = eigenvectors[:, idx]

# 合計人口が元の 1000人になるようにスケール調整
stable_population = stable_vector / np.sum(stable_vector) * 1000

print(f"最終的な人口（固有ベクトル）: 都市 {stable_population[0]:.1f}人, 地方 {stable_population[1]:.1f}人")