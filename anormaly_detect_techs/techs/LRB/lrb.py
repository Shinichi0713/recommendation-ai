import numpy as np
import cv2
from scipy import linalg
from scipy.spatial import distance
import matplotlib.pyplot as plt

def lrb_detection(img_rgb, win_out=9, win_in=3, k=20, lamda=0.1):
    """
    LRB (Local Representation-based Model) による異常検知
    
    Parameters:
    - k: 背景辞書から選択する近傍画素数 (K-Nearest Neighbors)
    - lamda: 正則化パラメータ
    """
    img_data = img_rgb.astype(np.float32)
    h, w, bands = img_data.shape
    r_out, r_in = win_out // 2, win_in // 2
    
    anomaly_map = np.zeros((h, w))
    pad_img = np.pad(img_data, ((r_out, r_out), (r_out, r_out), (0, 0)), mode='edge')
    
    # ガード窓マスク
    mask = np.ones((win_out, win_out), dtype=bool)
    mask[r_out-r_in : r_out+r_in+1, r_out-r_in : r_out+r_in+1] = False
    
    for i in range(h):
        for j in range(w):
            # 1. ターゲットベクトル y
            y = img_data[i, j, :].reshape(1, -1) # distance計算用に (1, Bands)
            
            # 2. 背景候補 D_all の抽出
            window = pad_img[i : i+win_out, j : j+win_out, :]
            D_all = window[mask]  # shape: (n_pixels, Bands)
            
            # 3. 【LRBの核心】距離計算とK個の選別
            # y と全背景画素のユークリッド距離を計算
            dists = distance.cdist(y, D_all, metric='euclidean').flatten()
            
            # 距離が近い順にインデックスを取得し、上位K個を選択
            idx_k = np.argsort(dists)[:k]
            D_k = D_all[idx_k].T  # shape: (Bands, K)
            
            # yを列ベクトルに戻す
            y_col = y.T
            
            # 4. 協調表現の計算 (選ばれた D_k を使用)
            # (D_k^T * D_k + lamda * I) alpha = D_k^T * y
            DktDk = np.dot(D_k.T, D_k)
            rhs = np.dot(D_k.T, y_col)
            
            try:
                # リッジ回帰を解く
                alpha, _, _, _ = linalg.lstsq(DktDk + lamda * np.eye(k), rhs)
                
                # 5. 再構成誤差
                y_hat = np.dot(D_k, alpha)
                anomaly_map[i, j] = np.linalg.norm(y_col - y_hat)
            except:
                anomaly_map[i, j] = 0
                
    return anomaly_map

# --- 実行と可視化 ---

# テスト用データの読み込み（適宜書き換えてください）
# img = cv2.imread('path_to_your_image.jpg')
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# サンプル生成: 緑の背景に小さな青い点
sample_img = np.zeros((60, 60, 3), dtype=np.uint8)
sample_img[:, :, 1] = 100 # 緑
sample_img[30:33, 30:33, 2] = 200 # 青い異常
sample_img = cv2.GaussianBlur(sample_img, (3,3), 0)

# LRB実行 (K=15程度でテスト)
print("Processing LRB... this may take a moment.")
lrb_score = lrb_detection(sample_img, win_out=9, win_in=5, k=15, lamda=0.1)

# 可視化
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.title("Original"); plt.imshow(sample_img)
plt.subplot(1, 2, 2); plt.title("LRB Score"); plt.imshow(lrb_score, cmap='jet'); plt.colorbar()
plt.show()