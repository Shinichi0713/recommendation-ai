import numpy as np
from sklearn.cluster import KMeans
from scipy import linalg
import matplotlib.pyplot as plt

def cluster_based_anomaly_detection(img_data, n_clusters=5, lamda=0.1):
    """
    クラスターベース辞書による異常検知
    
    Parameters:
    - img_data: (H, W, Bands) の画像データ
    - n_clusters: 背景をいくつのクラスターに分けるか
    - lamda: 正則化パラメータ
    """
    h, w, bands = img_data.shape
    X = img_data.reshape(-1, bands)  # 全画素を (N_pixels, Bands) に変換
    
    # --- 1. 構築フェーズ：背景のクラスタリング ---
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_  # 各クラスターの代表（重心）
    
    # 各クラスターごとの「プロジェクション行列（逆行列部分）」を事前計算して高速化
    # 辞書 D は各クラスターに属する画素から構成するが、
    # ここでは簡略化のためクラスター内の全画素を辞書として扱う
    projection_matrices = []
    cluster_dictionaries = []
    
    for c in range(n_clusters):
        # クラスター c に属する画素を抽出
        D_c = X[labels == c].T  # (Bands, n_samples_in_cluster)
        
        # 辞書が大きすぎる場合はサンプリングして計算負荷を抑える（例：最大500画素）
        if D_c.shape[1] > 500:
            idx = np.random.choice(D_c.shape[1], 500, replace=False)
            D_c = D_c[:, idx]
        
        # クラスター専用のプロジェクション行列 P = (D^T D + lambda*I)^-1 * D^T
        # alpha = P * y
        DtD = np.dot(D_c.T, D_c)
        I = np.eye(DtD.shape[0])
        P = np.dot(linalg.inv(DtD + lamda * I), D_c.T)
        
        projection_matrices.append(P)
        cluster_dictionaries.append(D_c)

    # --- 2. 検知フェーズ：再構成誤差の計算 ---
    print("Detecting anomalies...")
    anomaly_map_flat = np.zeros(h * w)
    
    for c in range(n_clusters):
        # クラスター c に属する画素のインデックスを取得
        idx = np.where(labels == c)[0]
        if len(idx) == 0: continue
        
        # ターゲット画素群 y
        Y_c = X[idx].T  # (Bands, n_pixels_in_this_cluster)
        
        # 重み alpha を一括計算: alpha = P * Y_c
        P = projection_matrices[c]
        alphas = np.dot(P, Y_c)
        
        # 再構成: Y_hat = D * alpha
        D = cluster_dictionaries[c]
        Y_hat = np.dot(D, alphas)
        
        # 各画素の再構成誤差（L2ノルム）を計算
        errors = np.linalg.norm(Y_c - Y_hat, axis=0)
        anomaly_map_flat[idx] = errors
        
    return anomaly_map_flat.reshape(h, w), labels.reshape(h, w)

# --- テスト実行用のダミーデータ生成 ---
h, w, b = 100, 100, 3
img = np.zeros((h, w, b))
img[:50, :, 0] = 0.5   # 上半分は赤い背景
img[50:, :, 1] = 0.5   # 下半分は緑の背景
img[25:30, 25:30, 2] = 1.0  # 赤い背景の中に青い異常
img[75:80, 75:80, 2] = 1.0  # 緑の背景の中に青い異常

# 実行
score_map, label_map = cluster_based_anomaly_detection(img, n_clusters=2)

# 可視化
plt.figure(figsize=(15, 5))
plt.subplot(1,3,1); plt.imshow(img); plt.title("Original Image")
plt.subplot(1,3,2); plt.imshow(label_map, cmap='tab10'); plt.title("Cluster Labels")
plt.subplot(1,3,3); plt.imshow(score_map, cmap='jet'); plt.title("Anomaly Score")
plt.show()