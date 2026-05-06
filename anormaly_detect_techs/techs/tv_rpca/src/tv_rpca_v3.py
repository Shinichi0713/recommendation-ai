import numpy as np
from scipy.fft import fft2, ifft2

def soft_thresholding(x, tau):
    """ソフトしきい値演算子"""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

def svt(x, tau):
    """特異値しきい値演算子 (Singular Value Thresholding)"""
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    s_thresholded = soft_thresholding(s, tau)
    return (u * s_thresholded) @ vh

def tv_rpca_admm(M, lambd=0.01, gamma=0.01, rho=1.0, max_iter=100, tol=1e-6):
    """
    TV-RPCA using ADMM (修正版)
    制約: L + S = M,  grad(L) - G = 0
    """
    rows, cols = M.shape
    
    # 変数の初期化
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    G = np.zeros((2, rows, cols))  # G[0]: x勾配, G[1]: y勾配
    Y1 = np.zeros_like(M)          # L + S = M 用の乗数
    Y2 = np.zeros((2, rows, cols)) # grad(L) - G = 0 用の乗数
    
    # FFTによるL更新用のカーネル準備
    # dx = [1, -1] (前方差分)
    dx = np.zeros((rows, cols))
    dx[0, 0] = 1; dx[0, 1] = -1
    dy = np.zeros((rows, cols))
    dy[0, 0] = 1; dy[1, 0] = -1
    
    # 分母: (I + grad^T grad) に対応する周波数応答
    # 随伴演算子 grad^T grad は離散ラプラシアンに対応
    denom = rho * (1 + np.abs(fft2(dx))**2 + np.abs(fft2(dy))**2)

    for i in range(max_iter):
        L_old = L.copy()
        
        # 1. S の更新
        ## argmin lambda|S|_1 + rho/2 ||M - L - S + Y1/rho||^2
        S = soft_thresholding(M - L + Y1/rho, lambd/rho)
        
        # 2. G の更新 (TV補助変数)
        ## argmin gamma|G|_1 + rho/2 ||grad(L) - G + Y2/rho||^2
        ## 以前のものは双対変数の符号が誤っていた
        grad_L_x = np.roll(L, -1, axis=1) - L
        grad_L_y = np.roll(L, -1, axis=0) - L
        
        ## 数式通りの + Y2/rho
        G[0] = soft_thresholding(grad_L_x + Y2[0]/rho, gamma/rho)
        G[1] = soft_thresholding(grad_L_y + Y2[1]/rho, gamma/rho)
        
        # 3. L の更新
        ## (I + grad^T grad) L = (M - S + Y1/rho) + grad^T (G - Y2/rho)
        
        ## grad^T (随伴演算子) は後方差分に対応
        diff_x = G[0] - Y2[0]/rho
        diff_y = G[1] - Y2[1]/rho
        
        ## 随伴演算子 nabla^T (G - Y2/rho) の計算
        div_G_Y = (np.roll(diff_x, 1, axis=1) - diff_x) + \
                  (np.roll(diff_y, 1, axis=0) - diff_y)
        
        ## 右辺 (RHS) の組み立て
        ## 注意: 数理的に R = rho*(M-S+Y1/rho) + rho*(div_G_Y) が正解
        R = rho * (M - S + Y1/rho) + rho * div_G_Y
        
        ## 最小二乗解を周波数領域で計算
        L_tilde = np.real(ifft2(fft2(R) / denom))
        
        ## 低ランク制約 (SVT) の適用
        L = svt(L_tilde, 1/rho)
        
        # 4. ラグランジュ乗数 Y1, Y2 の更新
        ## 制約の残差を積分
        res_Y1 = M - L - S
        Y1 += rho * res_Y1
        
        ## 最新のLを使って勾配を再計算
        grad_L_x_new = np.roll(L, -1, axis=1) - L
        grad_L_y_new = np.roll(L, -1, axis=0) - L
        Y2[0] += rho * (grad_L_x_new - G[0])
        Y2[1] += rho * (grad_L_y_new - G[1])
        
        ## 収束判定
        err = np.linalg.norm(L - L_old, 'fro') / np.linalg.norm(L_old, 'fro') if np.linalg.norm(L_old, 'fro') != 0 else 0
        if err < tol:
            break
            
    return L, S


# ダミーデータの生成
m, n = 64, 10  # 画像画素数 x 画像枚数
h = int(np.sqrt(m))
w = h
M_clean = np.random.randn(m, n)  # クリーンなデータ
M_noise = M_clean + 0.1 * np.random.randn(m, n)  # ガウスノイズ
M_sparse = np.zeros_like(M_noise)
M_sparse[np.random.rand(m, n) < 0.05] = 10.0  # スパースな外れ値
M = M_noise + M_sparse

# パラメータ設定
lambda_ = 1e-1 * 10  # SのL1正則化パラメータ
mu = 1.0      # TV正則化パラメータ

# RPCA+TVの実行
# tv_rpca_admm(M, lambd=0.01, gamma=0.01, rho=1.0, max_iter=100, tol=1e-6)
L, S = tv_rpca_admm(M, lambd=lambda_, gamma=0.00, rho=0.1, max_iter=50)

print("M shape:", M.shape)
print("L shape:", L.shape)
print("S shape:", S.shape)
print("再構成誤差:", np.linalg.norm(M - L - S, 'fro'))

# M, L, S を可視化
plot_matrix_as_images(M, '観測 M', h, w, n)
plot_matrix_as_images(L, '低ランク L', h, w, n)
plot_matrix_as_images(S, 'スパース S', h, w, n)