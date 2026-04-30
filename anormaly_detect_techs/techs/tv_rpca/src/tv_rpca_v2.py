import numpy as np
from scipy.fft import fft2, ifft2
from scipy.linalg import svd
from skimage.restoration import denoise_tv_chambolle  # TV用（参考）


def soft_threshold(x, tau):
    """要素ごとのソフト閾値作用素"""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

def svt(X, tau):
    """特異値閾値作用素（Singular Value Thresholding）"""
    U, s, Vh = svd(X, full_matrices=False)
    s_thresh = soft_threshold(s, tau)
    rank = np.sum(s_thresh > 0)
    return U[:, :rank] @ np.diag(s_thresh[:rank]) @ Vh[:rank, :]

def vector_soft_threshold(w, tau):
    """グループL1ノルムに対するベクトルソフト閾値作用素"""
    norm_w = np.sqrt(w[0]**2 + w[1]**2)
    scale = np.maximum(1.0 - tau / np.maximum(norm_w, 1e-10), 0.0)
    return w * scale[None, :, :]


def grad(u):
    """勾配演算子 D: u -> (Dx u, Dy u)"""
    u = u.astype(float)
    Dxu = np.roll(u, -1, axis=0) - u  # 前進差分（x方向）
    Dyu = np.roll(u, -1, axis=1) - u  # 前進差分（y方向）
    return np.stack([Dxu, Dyu], axis=0)

def div(p):
    """発散演算子 D^T: (px, py) -> D^T p"""
    px, py = p
    DxT_px = px - np.roll(px, 1, axis=0)   # 後退差分（x方向）
    DyT_py = py - np.roll(py, 1, axis=1)   # 後退差分（y方向）
    return DxT_px + DyT_py

def rpca_tv_admm(M, lambda_, mu, rho1=1.0, rho2=1.0, max_iter=100, tol=1e-4):
    """
    TV正則項付きRPCAをADMMで解く
    
    Parameters
    ----------
    M : ndarray, shape (m, n)
        観測行列（画像列：各列が1枚の画像）
    lambda_ : float
        SのL1正則化パラメータ
    mu : float
        TV正則化パラメータ
    rho1, rho2 : float
        ADMMのペナルティパラメータ
    max_iter : int
        最大反復回数
    tol : float
        収束判定の許容誤差
    
    Returns
    -------
    L : ndarray, shape (m, n)
        低ランク成分
    S : ndarray, shape (m, n)
        スパース成分
    """
    m, n = M.shape
    # 画像サイズ（ここでは sqrt(m) x sqrt(m) を仮定）
    h = int(np.sqrt(m))
    w = h
    assert h * w == m, "Mの行数は画像画素数である必要があります"
    
    # 変数の初期化
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    P = np.zeros((2, h, w, n))  # (2, h, w, n): (Dx L, Dy L) for each image
    Lambda = np.zeros_like(M)   # M = L + S の双対変数
    Gamma = np.zeros_like(P)    # P = D L の双対変数
    
    # 反復
    for it in range(max_iter):
        L_prev = L.copy()
        
        # --- L更新（核ノルム + TV） ---
        # 内側の二次最小化を簡単な反復で解く（ここでは1ステップの勾配降下で近似）
        # より正確には共役勾配法などを使うべき
        R1 = M - S + Lambda / rho1
        R2 = P + Gamma / rho2
        
        # 勾配降下のステップサイズ（経験的な値）
        alpha = 0.1 / (rho1 + rho2 * 8)  # 勾配演算子のノルムを考慮した調整
        
        for _ in range(5):  # 内側の反復回数（簡易）
            # 勾配の計算
            grad_L = rho1 * (L - R1) + rho2 * div(grad(L.reshape(h, w, n).transpose(2,0,1)) - R2)
            grad_L = grad_L.reshape(m, n)
            L = L - alpha * grad_L
        
        # 特異値閾値作用素を適用（核ノルム正則化）
        L = svt(L, 1.0 / (rho1 + rho2))
        
        # --- S更新（L1正則化） ---
        W_S = M - L + Lambda / rho1
        S = soft_threshold(W_S, lambda_ / rho1)
        
        # --- P更新（TV正則化） ---
        L_img = L.reshape(h, w, n).transpose(2,0,1)  # (n, h, w)
        W_P = np.stack([grad(L_img[i]) for i in range(n)], axis=-1)  # (2, h, w, n)
        W_P = W_P - Gamma / rho2
        P = vector_soft_threshold(W_P, mu / rho2)
        
        # --- 双対変数の更新 ---
        Lambda = Lambda + rho1 * (M - L - S)
        Gamma = Gamma + rho2 * (P - np.stack([grad(L_img[i]) for i in range(n)], axis=-1))
        
        # 収束判定
        diff_L = np.linalg.norm(L - L_prev, 'fro') / (np.linalg.norm(L_prev, 'fro') + 1e-10)
        if diff_L < tol:
            print(f"収束しました（反復回数: {it+1}）")
            break
    else:
        print(f"最大反復回数に達しました（{max_iter}回）")
    
    return L, S


if __name__ == "__main__":
    # ダミーデータの生成
    m, n = 64, 10  # 画像画素数 x 画像枚数
    h = int(np.sqrt(m))
    M_clean = np.random.randn(m, n)  # クリーンなデータ
    M_noise = M_clean + 0.1 * np.random.randn(m, n)  # ガウスノイズ
    M_sparse = np.zeros_like(M_noise)
    M_sparse[np.random.rand(m, n) < 0.05] = 10.0  # スパースな外れ値
    M = M_noise + M_sparse

    # パラメータ設定
    lambda_ = 0.1  # SのL1正則化パラメータ
    mu = 0.01      # TV正則化パラメータ

    # RPCA+TVの実行
    L, S = rpca_tv_admm(M, lambda_=lambda_, mu=mu, max_iter=50)

    print("M shape:", M.shape)
    print("L shape:", L.shape)
    print("S shape:", S.shape)
    print("再構成誤差:", np.linalg.norm(M - L - S, 'fro'))
