import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import svd

# 1. IALMによるRPCAの実装 (簡易版)
def ialm_rpca(M, lmbda=None, tol=1e-7, max_iter=100):
    if lmbda is None:
        lmbda = 1 / np.sqrt(np.max(M.shape))
    
    Y = M / np.max([np.linalg.norm(M, 2), np.linalg.norm(M, np.inf) / lmbda])
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    mu = 1.25 / np.linalg.norm(M, 2)
    rho = 1.5
    
    for i in range(max_iter):
        # Sの更新 (Soft Thresholding)
        temp_S = M - L + (1/mu) * Y
        S = np.sign(temp_S) * np.maximum(np.abs(temp_S) - lmbda/mu, 0)
        
        # Lの更新 (Singular Value Thresholding)
        u, s, vh = np.linalg.svd(M - S + (1/mu) * Y, full_matrices=False)
        s_thresholded = np.maximum(s - 1/mu, 0)
        L = np.dot(u * s_thresholded, vh)
        
        # 残差の更新
        Z = M - L - S
        Y = Y + mu * Z
        mu = mu * rho
        
        if np.linalg.norm(Z, 'fro') / np.linalg.norm(M, 'fro') < tol:
            break
    return L, S

# 2. RX Detectorの実装
def rx_detector(S_matrix):
    mu = np.mean(S_matrix, axis=0)
    sigma = np.cov(S_matrix, rowvar=False) + 1e-6 * np.eye(S_matrix.shape[1])
    sigma_inv = np.linalg.inv(sigma)
    
    diff = S_matrix - mu
    # まとめて行列演算でマハラノビス距離を計算
    scores = np.sum(np.dot(diff, sigma_inv) * diff, axis=1)
    return scores

# --- メイン処理 ---

# 画像の読み込み (グレースケール)
img = cv2.imread('anomaly_sample.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    # サンプル画像がない場合のダミーデータ生成 (100x100)
    img = np.ones((100, 100)) * 128
    img[45:55, 45:55] = 200 # 異常を模した白い矩形
    img = img + np.random.normal(0, 5, img.shape) # ノイズ

# パッチ分割設定 (例: 10x10パッチ)
p_size = 10
h, w = img.shape
patches = []

for i in range(0, h - p_size + 1, p_size):
    for j in range(0, w - p_size + 1, p_size):
        patch = img[i:i+p_size, j:j+p_size].flatten()
        patches.append(patch)

M = np.array(patches).astype(np.float32)

# RPCAの実行
L_hat, S_hat = ialm_rpca(M)

# RX Detectorの実行 (スパース成分 S に対して)
rx_scores = rx_detector(S_hat)

# 可視化用に元の形状に戻す
rx_map = rx_scores.reshape((h // p_size, w // p_size))
rx_map_resized = cv2.resize(rx_map, (w, h), interpolation=cv2.INTER_NEAREST)

# 結果表示
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1); plt.imshow(img, cmap='gray'); plt.title("Original")
plt.subplot(1, 4, 2); plt.imshow(L_hat.reshape(M.shape).reshape(-1, p_size, p_size)[0], cmap='gray'); plt.title("L component (Patch 0)")
plt.subplot(1, 4, 3); plt.imshow(np.abs(S_hat).mean(axis=1).reshape(rx_map.shape), cmap='hot'); plt.title("S intensity")
plt.subplot(1, 4, 4); plt.imshow(rx_map, cmap='jet'); plt.title("RX Score Map")
plt.show()