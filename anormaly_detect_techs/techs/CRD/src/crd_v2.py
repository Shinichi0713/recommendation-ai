import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

def crd_detector(img, win_out=9, win_in=3, lamda=0.01):
    """
    Collaborative Representation-based Detector (CRD)
    
    Parameters:
    - img: (H, W, Bands) の画像配列
    - win_out: 外窓（背景辞書）のサイズ
    - win_in: 内窓（ガード窓）のサイズ
    - lamda: 正則化パラメータ (Ridge回帰の係数)
    """
    h, w, bands = img.shape
    anomaly_map = np.zeros((h, w))
    
    # 境界パディング（窓処理のため）
    pad_size = win_out // 2
    img_pad = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
    
    # 単位行列（正則化用）を事前に作成
    # 窓内の画素数からガード窓の画素数を引いたものが辞書の数
    n_dict = win_out**2 - win_in**2
    I = np.eye(n_dict)

    print("Processing CRD...")
    for r in range(h):
        for c in range(w):
            # 1. 窓領域の抽出
            # パディング済み画像から現在のピクセルを中心とした窓を切り出す
            r_idx, c_idx = r + pad_size, c + pad_size
            local_win = img_pad[r_idx-pad_size:r_idx+pad_size+1, c_idx-pad_size:c_idx+pad_size+1, :]
            
            # 2. 背景辞書 D の作成（ガード窓を除去）
            # 窓の中心部分（win_in）を除いた画素を並べて行列Dを作る
            mask = np.ones((win_out, win_out), dtype=bool)
            in_start = (win_out - win_in) // 2
            mask[in_start:in_start+win_in, in_start:in_start+win_in] = False
            
            D = local_win[mask].T  # (Bands, n_dict)
            y = img[r, c, :].reshape(-1, 1) # ターゲットピクセル (Bands, 1)
            
            # 3. 係数ベクトル alpha の推定 (Ridge Regression)
            # alpha = inv(D.T @ D + lambda * I) @ D.T @ y
            # 数値的安定性のために D.T @ D を計算
            DtD = D.T @ D
            try:
                alpha = inv(DtD + lamda * I) @ D.T @ y
                
                # 4. 再構成誤差（異常スコア）の算出
                # Score = || y - D @ alpha ||^2
                reconstructed_y = D @ alpha
                score = np.linalg.norm(y - reconstructed_y)**2
                anomaly_map[r, c] = score
            except:
                anomaly_map[r, c] = 0
                
    return anomaly_map

# --- 1. テストデータの生成 (Colabで確認用) ---
# 背景: 青い空(上半分)と緑の草原(下半分)の境界
test_img = np.zeros((60, 60, 3))
test_img[:30, :, :] = [0.5, 0.7, 1.0]  # Sky
test_img[30:, :, :] = [0.2, 0.6, 0.2]  # Grass
test_img += np.random.normal(0, 0.02, test_img.shape) # ノイズ

# 異常: 境界付近に置かれた小さな赤い点
test_img[28:31, 30:33, :] = [1.0, 0.0, 0.0]

# --- 2. 実行 ---
# lambdaを小さくしすぎると背景を再現しすぎて異常が消えるため、0.01〜1程度で調整
scores = crd_detector(test_img, win_out=7, win_in=3, lamda=0.1)

# --- 3. 可視化 ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(test_img)
plt.subplot(1, 2, 2)
plt.title("CRD Anomaly Score")
plt.imshow(scores, cmap='jet')
plt.colorbar()
plt.show()