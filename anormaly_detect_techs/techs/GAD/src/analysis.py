import networkx as nx
import numpy as np
import cv2
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def visualize_graph_structure(img_section, n_neighbors=5):
    h, w, b = img_section.shape
    X = img_section.reshape(-1, b)
    
    # 1. グラフ作成
    W = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance')
    G = nx.from_scipy_sparse_array(W)
    
    # 2. 座標の設定（ピクセルの位置をそのまま座標にする）
    pos = {i: (i % w, h - 1 - (i // w)) for i in range(h * w)}
    
    # 3. 描画
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, node_size=50, node_color='skyblue', edge_color='gray', alpha=0.6)
    plt.title("Local Pixel Network (Spatial Grid)")
    plt.show()

# --- テストデータの作成と実行 ---
# ダミー画像（32x32ピクセル、3バンド）
img = np.random.normal(0, 0.1, (32, 32, 3))
# 異常（背景の分布から外れた色を持つ小さな島）
img[15:18, 15:18, :] += 2.0

# 小さなパッチで実行
visualize_graph_structure(img[13:23, 13:23, :])


def visualize_spectral_embedding(eigenvectors, labels):
    # eigenvectors: [n_pixels, n_components]
    # labels: 異常か背景かを示すフラグ（可視化用）
    
    plt.figure(figsize=(8, 6))
    plt.scatter(eigenvectors[:, 1], eigenvectors[:, 2], c=labels, cmap='coolwarm', s=10, alpha=0.5)
    plt.xlabel("Eigenvector 2")
    plt.ylabel("Eigenvector 3")
    plt.title("Spectral Embedding Space")
    plt.colorbar(label="Anomaly Score")
    plt.grid(True)
    plt.show()

# 実行（前述のgad_detector内のfeaturesを使用）
visualize_spectral_embedding(eigenvectors, score_gad)