import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
from scipy.linalg import inv

def segmentation_based_rx(img, n_segments=150, compactness=10, regularization=1e-5):
    """
    Segmentation-based RX Detector
    
    Parameters:
    - img: (H, W, Bands) の numpy 配列
    - n_segments: 分割するセグメント（スーパーピクセル）の概数
    - compactness: 色の近さと空間的な近さのバランス（高いほど正方形に近くなる）
    - regularization: 共分散行列の逆行列計算を安定させるための正則化パラメータ
    """
    h, w, bands = img.shape
    
    # 1. セグメンテーション (SLICアルゴリズム)
    # 道路や草地といった「塊」を作る
    segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)
    
    # スコア格納用の配列
    anomaly_map = np.zeros((h, w))
    
    # 2. セグメントごとにループして統計計算
    unique_labels = np.unique(segments)
    print(f"Total segments to process: {len(unique_labels)}")
    
    for label in unique_labels:
        # セグメントに属する画素のマスク
        mask = (segments == label)
        pixels = img[mask] # (N_pixels_in_segment, Bands)
        
        # 画素数がバンド数より少ないと共分散行列が計算できないためスキップ
        if len(pixels) <= bands:
            continue
            
        # --- 背景統計量の計算 ---
        # セグメント内の平均ベクトル
        mu = np.mean(pixels, axis=0)
        
        # セグメント内の共分散行列
        # (x - mu).T @ (x - mu) / (N - 1)
        diff = pixels - mu
        sigma = (diff.T @ diff) / (len(pixels) - 1)
        
        # 数値的安定化のための正則化 (対角成分に微小値を足す)
        sigma += np.eye(bands) * regularization
        
        # --- 異常スコアの算出 (マハラノビス距離) ---
        try:
            sigma_inv = inv(sigma)
            # このセグメントに属する各画素について計算
            # 効率化のため、セグメント単位で行列演算
            scores = np.einsum('ij,jk,ik->i', diff, sigma_inv, diff)
            anomaly_map[mask] = scores
        except np.linalg.LinAlgError:
            # 行列が特異（逆行列が計算不能）な場合はスキップ
            continue
            
    return anomaly_map, segments

# --- テスト用の擬似データ作成 ---
# 背景: 草原(緑)と道路(グレー)の境界を持つ画像
img_test = np.zeros((100, 100, 3))
img_test[:, :50, :] = [0.2, 0.5, 0.2]  # 草原 (Green)
img_test[:, 50:, :] = [0.4, 0.4, 0.4]  # 道路 (Gray)
# ノイズ追加
img_test += np.random.normal(0, 0.02, img_test.shape)
# 異常: 道路の上に置かれた小さな赤い物体
img_test[70:75, 70:75, :] = [0.9, 0.1, 0.1] 

# 実行
score_map, seg_labels = segmentation_based_rx(img_test, n_segments=50)

# 可視化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_test)
axes[0].set_title("Original Scene (Grass & Road)")
axes[1].imshow(label2rgb(seg_labels, img_test, kind='avg'))
axes[1].set_title("Segments (Superpixels)")
axes[2].imshow(score_map, cmap='hot')
axes[2].set_title("Anomaly Score (Seg-RX)")
plt.show()