import numpy as np
import matplotlib.pyplot as plt

def svt(M, tau):
    """特異値しきい値演算 (Singular Value Thresholding)"""
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    # 特異値を tau でソフトしきい値処理
    S_thresh = np.maximum(S - tau, 0)
    return np.dot(U * S_thresh, Vh)

def soft_threshold(x, lam):
    """ソフトしきい値関数"""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def ialm_rpca(M, max_iter=100, tol=1e-7):
    # 行列のサイズ
    m, n = M.shape
    
    # パラメータ設定 (標準的な推移ルール)
    lam = 1.0 / np.sqrt(max(m, n))
    mu = 1.25 / np.linalg.norm(M, ord=2) # 初期mu
    rho = 1.5 # muの更新倍率
    
    # 変数の初期化
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = np.zeros_like(M) # ラグランジュ乗数
    
    # 履歴保存用
    history = {
        'residual': [],      # ||M - L - S||_F / ||M||_F
        'inner_prod': [],    # <Y, M - L - S>
        'penalty_term': []   # (mu/2) * ||M - L - S||_F^2
    }
    
    m_norm = np.linalg.norm(M, 'fro')
    
    for i in range(max_iter):
        # 1. 背景 L の更新 (SVTを使用)
        L = svt(M - S + (1/mu) * Y, 1/mu)
        
        # 2. 異常 S の更新 (ソフトしきい値を使用)
        S = soft_threshold(M - L + (1/mu) * Y, lam/mu)
        
        # 3. ズレ（残差）の計算
        Z = M - L - S
        z_norm = np.linalg.norm(Z, 'fro')
        
        # 履歴の記録
        # 項1: <Y, M - L - S>
        history['inner_prod'].append(np.sum(Y * Z))
        # 項2: (mu/2) * ||M - L - S||_F^2
        history['penalty_term'].append((mu / 2.0) * (z_norm**2))
        # 全体残差（収束判定用）
        history['residual'].append(z_norm / m_norm)
        
        # 4. ラグランジュ乗数 Y の更新
        Y = Y + mu * Z
        
        # 5. ペナルティ係数 mu の更新
        mu = mu * rho
        
        # 収束判定
        if history['residual'][-1] < tol:
            print(f"Converged at iteration {i}")
            break
            
    return L, S, history

# --- 1. テストデータの生成 (修正版) ---
np.random.seed(42)
n = 50
# 低ランク行列 L (背景: ランク2)
L_true = np.dot(np.random.randn(n, 2), np.random.randn(2, n))

# スパース行列 S (異常)
S_true = np.zeros((n, n))
mask = np.random.rand(n, n) < 0.05 # 5%のマスクを作成
S_true[mask] = np.random.randn(np.sum(mask)) # マスクされた数だけ乱数を生成

# 観測行列 M
M = L_true + S_true

# --- 2. IALMの実行 ---
L_res, S_res, hist = ialm_rpca(M)

# --- 3. 可視化 ---
plt.figure(figsize=(15, 5))

# 残差の推移
plt.subplot(1, 2, 1)
plt.plot(hist['residual'], label='Relative Residual (||M-L-S||/||M||)', color='black', linewidth=2)
plt.yscale('log')
plt.xlabel('Iteration')
plt.title('Convergence (Log Scale)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# 各項の推移
plt.subplot(1, 2, 2)
plt.plot(hist['inner_prod'], label='Inner Product <Y, Z>', color='red')
plt.plot(hist['penalty_term'], label='Penalty Term (mu/2)||Z||^2', color='blue', linestyle='--')
plt.xlabel('Iteration')
plt.title('Lagrangian Terms comparison')
plt.legend()

plt.tight_layout()
plt.show()