import numpy as np
import cv2
from scipy.spatial import distance
import matplotlib.pyplot as plt

def lrx_detector(img_data, win_out=9, win_in=3, reg_factor=1e-6):
    """
    LRX (Local Reed-Xiaoli) 異常検知アルゴリズム
    
    Parameters:
    - img_data: (H, W, Bands) の画像配列
    - win_out: 外窓のサイズ（奇数）
    - win_in: 内窓（ガード窓）のサイズ（奇数）
    - reg_factor: 共分散行列の正則化パラメータ（特異行列対策）
    """
    h, w, bands = img_data.shape
    r_out = win_out // 2
    r_in = win_in // 2
    
    # 異常スコアを格納するマップ
    anomaly_map = np.zeros((h, w))
    
    # パディング処理
    pad_img = np.pad(img_data, ((r_out, r_out), (r_out, r_out), (0, 0)), mode='edge')
    
    # ガード窓用のマスク作成（背景画素だけを抽出するため）
    mask = np.ones((win_out, win_out), dtype=bool)
    mask[r_out-r_in : r_out+r_in+1, r_out-r_in : r_out+r_in+1] = False
    
    # 全ピクセルをループ（高速化が必要な場合は後述のヒント参照）
    for i in range(h):
        for j in range(w):
            # 1. ターゲットピクセル y (Bands,)
            y = img_data[i, j, :]
            
            # 2. 局所背景窓の抽出 (n_pixels, Bands)
            window = pad_img[i : i+win_out, j : j+win_out, :]
            background_pixels = window[mask] # ガード窓を除外
            
            # 3. 背景の統計量（平均と共分散）を計算
            mu = np.mean(background_pixels, axis=0)
            # 背景から平均を引く
            diff_bg = background_pixels - mu
            # 共分散行列 Σ (Bands x Bands)
            sigma = (diff_bg.T @ diff_bg) / (len(background_pixels) - 1)
            
            # 4. 正則化（逆行列を計算可能にするため対角成分に微小値を加算）
            sigma += np.eye(bands) * reg_factor
            
            # 5. マハラノビス距離の計算
            # Score = (y - mu)^T * Σ^-1 * (y - mu)
            try:
                diff_y = (y - mu).reshape(-1, 1)
                # 逆行列を直接求めるより solve を使う方が安定
                score = diff_y.T @ np.linalg.solve(sigma, diff_y)
                anomaly_map[i, j] = score
            except np.linalg.LinAlgError:
                anomaly_map[i, j] = 0
                
    return anomaly_map

# --- 実行例 ---
# 3バンドのダミーデータ生成
img = np.random.normal(0, 0.1, (50, 50, 3))
img[20:25, 20:25, :] += 1.5 # 異常（大きな塊）
img[40, 40, :] += 2.0        # 異常（点）

# LRX実行
# 大きな異常を検知するために win_in を大きめに設定 (例: 7)
score_lrx = lrx_detector(img, win_out=15, win_in=7)

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1); plt.imshow(img[:,:,0]); plt.title("Original (Band 0)")
plt.subplot(1,2,2); plt.imshow(score_lrx, cmap='jet'); plt.title("LRX Score")
plt.colorbar(); plt.show()