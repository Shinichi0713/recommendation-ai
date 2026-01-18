import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def crd_anomaly_detection(image_gray, win_out=9, win_in=3, lamda=0.01):
    """
    Collaborative Representation-based Detector (CRD)
    
    Parameters:
    - image_gray: グレースケール画像 (numpy array)
    - win_out: 背景辞書を作成する外側の窓サイズ (例: 9x9)
    - win_in:  ターゲットと干渉しないための内側のガード窓サイズ (例: 3x3)
    - lamda:   正則化パラメータ (L2ノルムの強さ)
    """
    h, w = image_gray.shape
    # スコアを格納する行列
    anomaly_map = np.zeros((h, w))
    
    # 窓の半径を計算
    r_out = win_out // 2
    r_in = win_in // 2
    
    # パディング（端の画素も処理できるようにする）
    img_pad = np.pad(image_gray, r_out, mode='edge')
    
    # 全画素をスキャン
    for i in range(r_out, h + r_out):
        for j in range(r_out, w + r_out):
            # --- 1. 背景辞書の構築 ---
            # 外側の窓を切り出す
            local_win = img_pad[i-r_out:i+r_out+1, j-r_out:j+r_out+1]
            
            # 内側の窓（ガード窓）をマスクして除外
            mask = np.ones((win_out, win_out), dtype=bool)
            mask[r_out-r_in:r_out+r_in+1, r_out-r_in:r_out+r_in+1] = False
            
            # 背景画素を抽出して辞書 D (ベクトル並び) に変換
            D = local_win[mask].reshape(-1, 1) # [辞書の要素数, 1]
            
            # --- 2. ターゲット画素の取得 ---
            y = img_pad[i, j] # 中心の一点
            
            # --- 3. 協調表現による再構成 (リッジ回帰) ---
            # 重み係数 alpha = (D^T * D + lamda * I)^-1 * D^T * y
            # 今回は各画素がスカラー(1次元)の場合、以下の簡略式で計算可能
            # 誤差 E = |y - D * alpha|^2
            
            # 行列演算の安定化のため D^T * D を計算
            DtD = np.dot(D.T, D)
            inv_part = 1.0 / (DtD + lamda)
            alpha = inv_part * np.dot(D.T, y)
            
            # 再構成誤差の算出
            y_hat = np.dot(D, alpha)
            # 再構成した各背景画素との差の最小値、または代表値との誤差
            # ここではシンプルに中心点yと再構成近似値の差をスコアとする
            # (数式上の最小二乗誤差をスカラー的に解釈)
            residual = np.abs(y - np.mean(y_hat))
            
            anomaly_map[i-r_out, j-r_out] = residual
            
    return anomaly_map

# --- 実行プロセス ---

# 画像の読み込み（ここではサンプルとして適当な画像を生成、または読み込み）
# img = cv2.imread('test_sample.jpg', 0) 
img = np.full((100, 100), 128, dtype=np.float32) # グレーの背景
img[40:45, 40:45] = 255 # 白い異常点（キズ）を模擬
img += np.random.normal(0, 5, img.shape) # 少しのノイズ

# CRDの実行
score_map = crd_anomaly_detection(img, win_out=11, win_in=5, lamda=0.1)

# 結果の可視化
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title("Original Image")
plt.subplot(1, 3, 2); plt.imshow(score_map, cmap='jet'); plt.title("CRD Anomaly Score")
plt.subplot(1, 3, 3); plt.imshow(score_map > np.percentile(score_map, 99), cmap='gray'); plt.title("Thresholded Result")
plt.show()