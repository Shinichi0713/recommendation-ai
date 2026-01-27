import numpy as np
from scipy import linalg
from scipy.spatial import distance
import matplotlib.pyplot as plt

def lrb_multispectral(img_data, win_out=9, win_in=3, k=20, lamda=0.1, metric='cosine'):
    """
    マルチスペクトル/ハイパースペクトルデータ用 LRB
    
    Parameters:
    - img_data: (H, W, Bands) の numpy 配列
    - k: 選択する近傍スペクトル数
    - metric: 距離尺度 ('cosine' はスペクトルの形状を重視, 'euclidean' は強度も重視)
    """
    h, w, bands = img_data.shape
    r_out, r_in = win_out // 2, win_in // 2
    
    anomaly_map = np.zeros((h, w))
    
    # パディング（全バンド一括）
    pad_img = np.pad(img_data, ((r_out, r_out), (r_out, r_out), (0, 0)), mode='edge')
    
    # 背景抽出用マスク
    mask = np.ones((win_out, win_out), dtype=bool)
    mask[r_out-r_in : r_out+r_in+1, r_out-r_in : r_out+r_in+1] = False
    
    for i in range(h):
        for j in range(w):
            # 1. ターゲットスペクトル y (1, Bands)
            y = img_data[i, j, :].reshape(1, -1)
            
            # 2. 背景候補スペクトル群 D_all (n_pixels, Bands)
            window = pad_img[i : i+win_out, j : j+win_out, :]
            D_all = window[mask] 
            
            # 3. スペクトルの類似度に基づき K 個選別
            # マルチスペクトルではコサイン距離が一般的に有効（波長パターンの比較）
            dists = distance.cdist(y, D_all, metric=metric).flatten()
            idx_k = np.argsort(dists)[:k]
            D_k = D_all[idx_k].T  # (Bands, K)
            
            y_col = y.T # (Bands, 1)
            
            # 4. 協調表現 (リッジ回帰)
            # A = (D_k^T * D_k + lamda * I)
            # b = D_k^T * y
            DktDk = np.dot(D_k.T, D_k)
            rhs = np.dot(D_k.T, y_col)
            
            try:
                # 最小二乗解 alpha を求める
                alpha, _, _, _ = linalg.lstsq(DktDk + lamda * np.eye(k), rhs)
                
                # 5. 再構成誤差を異常スコアとする
                y_hat = np.dot(D_k, alpha)
                anomaly_map[i, j] = np.linalg.norm(y_col - y_hat)
            except linalg.LinAlgError:
                anomaly_map[i, j] = 0
                
    return anomaly_map


if __name__ == "__main__":
    # テスト用のダミーデータ生成
    H, W, Bands = 50, 50, 10
    np.random.seed(0)
    img_data = np.random.rand(H, W, Bands)
    
    # LRB 実行
    anomaly_map = lrb_multispectral(img_data, win_out=9, win_in=3, k=15, lamda=0.1, metric='cosine')
    
    # 結果表示
    plt.imshow(anomaly_map, cmap='hot')
    plt.colorbar()
    plt.title('Anomaly Map from LRB')
    plt.show()