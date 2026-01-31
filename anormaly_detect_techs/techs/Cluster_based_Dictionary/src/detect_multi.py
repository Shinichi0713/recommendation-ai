import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import linalg
import matplotlib.pyplot as plt

def cluster_based_msi_detector(img_data, n_clusters=5, pca_components=0.99, lamda=0.1):
    """
    マルチスペクトル画像用 クラスターベース異常検知
    
    Parameters:
    - img_data: (H, W, Bands) のマルチスペクトルデータ
    - n_clusters: クラスター数
    - pca_components: PCAで保持する累積寄与率（または成分数）
    - lamda: リッジ回帰の正則化パラメータ
    """
    h, w, bands = img_data.shape
    X_raw = img_data.reshape(-1, bands)
    
    # 1. 前処理：各バンドの正規化（スケールを揃える）
    X_norm = (X_raw - np.mean(X_raw, axis=0)) / (np.std(X_raw, axis=0) + 1e-6)
    
    # 2. 次元圧縮 (PCA): クラスタリングの安定化と高速化のため
    # 99%の情報を保持しつつ、ノイズ成分をカット
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_norm)
    print(f"PCA reduced bands from {bands} to {X_pca.shape[1]}")

    # 3. 背景のクラスタリング
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    anomaly_map_flat = np.zeros(h * w)
    
    # 4. クラスターごとの再構成（オリジナルの高次元空間で計算）
    for c in range(n_clusters):
        # クラスター c に属する画素のインデックス
        idx = np.where(labels == c)[0]
        if len(idx) == 0: continue
        
        # クラスター内の全スペクトル (Bands, N_samples)
        # 実際にはここから辞書 D_c をサンプリングして構築
        D_candidates = X_raw[idx].T 
        
        # 辞書サイズが大きすぎると逆行列が重いため制限
        n_dict = min(D_candidates.shape[1], 1000)
        sample_idx = np.random.choice(D_candidates.shape[1], n_dict, replace=False)
        D_c = D_candidates[:, sample_idx]
        
        # ターゲットスペクトル群 y (Bands, N_targets)
        Y_c = X_raw[idx].T
        
        # --- 協調表現による再構成計算 ---
        # alpha = (D^T D + lambda*I)^-1 * D^T * y
        DtD = np.dot(D_c.T, D_c)
        I = np.eye(n_dict)
        # 逆行列部分を事前計算
        P = np.dot(linalg.inv(DtD + lamda * I), D_c.T)
        
        # 重みと再構成
        alphas = np.dot(P, Y_c)
        Y_hat = np.dot(D_c, alphas)
        
        # 各ピクセルの再構成誤差をスコア化
        # スペクトル形状の差をみるためL2ノルムを使用
        errors = np.linalg.norm(Y_c - Y_hat, axis=0)
        anomaly_map_flat[idx] = errors

    return anomaly_map_flat.reshape(h, w), labels.reshape(h, w)

# --- テスト用のダミーデータ（10バンド） ---
data = np.random.normal(0, 0.1, (50, 50, 10))
data[10:40, 10:40, :] += 0.5  # 背景クラス1
data[30:35, 30:35, 5] += 2.0  # 異常（第5バンドだけ異常に強い）

# 実行
score_map, cluster_map = cluster_based_msi_detector(data, n_clusters=2)

# 可視化
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1); plt.imshow(data[:,:,[0,1,2]]); plt.title("RGB Composite (Pseudo)")
plt.subplot(1,3,2); plt.imshow(cluster_map, cmap='Set3'); plt.title("Clusters")
plt.subplot(1,3,3); plt.imshow(score_map, cmap='hot'); plt.title("Anomaly Score")
plt.show()