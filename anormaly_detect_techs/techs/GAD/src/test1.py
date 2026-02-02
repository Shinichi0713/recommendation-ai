import numpy as np
import cv2
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def gad_detector(img_data, n_neighbors=15, n_components=3):
    """
    GAD (Graph-based Anomaly Detector) の簡易実装
    
    Parameters:
    - img_data: (H, W, Bands) の画像配列
    - n_neighbors: 各ピクセルからエッジを張る近傍数（k-NN）
    - n_components: 解析に使用する固有ベクトルの数
    """
    h, w, bands = img_data.shape
    n_pixels = h * w
    
    # 1. データを (画素数, バンド数) の行列に変換
    X = img_data.reshape(n_pixels, bands)
    
    # 2. グラフの構築 (k-Nearest Neighbors Graph)
    # 各ピクセルとその色の近い近傍をエッジで結ぶ
    print("Building graph...")
    # 疎行列（Sparse Matrix）として隣接行列 W を作成
    W = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=False)
    
    # 距離を親和性（重み）に変換: exp(-d^2 / sigma^2)
    # ここでは簡略化のため、距離の逆数等で重み付け
    W.data = np.exp(-W.data**2 / (2 * np.mean(W.data)**2))
    
    # 3. グラフ・ラプラシアン L の計算
    # L = D - W (Dは次数行列)
    print("Computing Laplacian...")
    L = laplacian(W, normed=True)
    
    # 4. 固有値分解 (Spectral Embedding)
    # 最小の固有値（0付近）に対応する固有ベクトルを抽出
    # 最初の固有ベクトルは背景の連結成分を示すため、2番目以降を使用
    print("Eigenvalue decomposition...")
    eigenvalues, eigenvectors = eigsh(L, k=n_components+1, which='SM')
    
    # 5. 異常スコアの算出
    # 固有ベクトルの空間（低次元空間）に写像した際の、原点または平均からの距離
    # 背景は特定の場所に固まり、異常はそこから外れる
    features = eigenvectors[:, 1:] # 最初の固有ベクトルを除く
    mean_feat = np.mean(features, axis=0)
    scores = np.linalg.norm(features - mean_feat, axis=1)
    
    # 2次元のマップに戻す
    anomaly_map = scores.reshape(h, w)
    
    return anomaly_map

# --- テストデータの作成と実行 ---
# ダミー画像（32x32ピクセル、3バンド）
img = np.random.normal(0, 0.1, (32, 32, 3))
# 異常（背景の分布から外れた色を持つ小さな島）
img[15:18, 15:18, :] += 2.0

# GAD実行
score_gad = gad_detector(img, n_neighbors=20, n_components=3)

# 可視化
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img[:, :, 0], cmap='gray')
plt.title("Original Band 0")
plt.subplot(1, 2, 2)
plt.imshow(score_gad, cmap='hot')
plt.colorbar()
plt.title("GAD Anomaly Score")
plt.show()