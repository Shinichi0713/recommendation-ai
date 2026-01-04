import numpy as np
import matplotlib.pyplot as plt

def soft_threshold(x, lam):
    """ソフトしきい値関数"""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def robust_dictionary_learning(y, n_atoms, iterations=100, lam_x=0.1, lam_s=0.5):
    """
    RDLの簡易実装 (辞書更新は簡略化のため逐次更新)
    y: 観測データ (dim, n_samples)
    n_atoms: 辞書のサイズ
    """
    dim, n_samples = y.shape
    # 初期化
    D = np.random.randn(dim, n_atoms)
    D /= np.linalg.norm(D, axis=0) # 列正規化
    x = np.zeros((n_atoms, n_samples))
    S = np.zeros((dim, n_samples))
    
    alpha = 0.01 # 学習率

    for i in range(iterations):
        # 1. Sparse Coding (xの更新): 異常Sを除いた残差に対してISTA的な更新
        for _ in range(10):
            residual_x = (y - S) - np.dot(D, x)
            x = soft_threshold(x + alpha * np.dot(D.T, residual_x), lam_x)

        # 2. Anomaly Detection (Sの更新): 背景Dxを除いた残差から異常を抽出
        residual_s = y - np.dot(D, x)
        S = soft_threshold(residual_s, lam_s)

        # 3. Dictionary Update (Dの更新): 背景をよりよく表現するようにDを修正
        # (簡単のため勾配降下法)
        D += alpha * np.dot((y - S - np.dot(D, x)), x.T)
        D /= np.linalg.norm(D, axis=0) + 1e-8 # 正規化

    return D, x, S

# --- 1. デモデータの作成 ---
np.random.seed(42)
t = np.linspace(0, 1, 100)
# 正常なパターン: サイン波の組み合わせ
base_pattern = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
data = np.tile(base_pattern, (20, 1)).T # 20サンプル分コピー
data += np.random.randn(*data.shape) * 0.05 # 微小なノイズ

# 異常の注入: 特定の箇所に強いスパイク（キズ）を入れる
data[40:45, 5] += 3.0
data[70:75, 12] -= 2.5

# --- 2. RDLの実行 ---
# n_atoms=3 (少ないパーツで背景を説明させる)
D_res, x_res, S_res = robust_dictionary_learning(data, n_atoms=3)

# --- 3. 結果の可視化 ---
sample_idx = 5 # 異常を入れたサンプルを表示
plt.figure(figsize=(15, 8))

plt.subplot(4, 1, 1)
plt.plot(data[:, sample_idx], label='Original (Background + Anomaly)', color='black')
plt.title("Input Signal (Observation)")
plt.legend()

plt.subplot(4, 1, 2)
background = np.dot(D_res, x_res)
plt.plot(background[:, sample_idx], label='Recovered Background (Dx)', color='blue')
plt.title("Learned Background (Low-rank component)")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(S_res[:, sample_idx], label='Detected Anomaly (S)', color='red')
plt.title("Extracted Sparse Anomaly")
plt.legend()

plt.subplot(4, 1, 4)
plt.imshow(D_res, aspect='auto', cmap='viridis')
plt.title("Learned Dictionary Atoms (The 'Building Blocks')")

plt.tight_layout()
plt.show()