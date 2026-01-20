import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import linalg


def detect_crd(img_data, win_out=9, win_in=3, lamda=1e-4):
    """
    Detects anomalies in an image using a hypothetical CRD technique.

    Args:
        img_data (numpy.ndarray): Input image data.
        win_out (int): Outer window size.
        win_in (int): Inner window size.
        lamda (float): Regularization parameter.
        threshold (float): Threshold for anomaly detection.
    """

    h, w, bands = img_data.shape
    r_out = win_out // 2
    r_in = win_in // 2

    # Initialize anomaly score map
    anomaly_map = np.zeros((h, w))

    # Pad image for boundary handling
    pad_img = np.pad(img_data, ((r_out, r_out), (r_out, r_out), (0, 0)), mode='edge')

    # Create mask for guard window (True for background pixels)
    mask = np.ones((win_out, win_out), dtype=bool)
    mask[r_out-r_in : r_out+r_in+1, r_out-r_in : r_out+r_in+1] = False

    # Sliding window processing
    for i in range(h):
        for j in range(w):
            # 1. Target vector y: (Bands, 1)
            y = img_data[i, j, :].reshape(-1, 1)

            # 2. Construct background dictionary D: (Bands, n_background_pixels)
            window = pad_img[i : i+win_out, j : j+win_out, :]
            D = window[mask].T  # shape: (Bands, n)

            # 3. Compute collaborative representation (ridge regression solution)
            DtD = np.dot(D.T, D)
            n_pixels = D.shape[1]

            alpha = linalg.solve(DtD + lamda * np.eye(n_pixels), np.dot(D.T, y), assume_a='pos')

            # 4. Reconstruction and error (anomaly score) calculation
            y_hat = np.dot(D, alpha)
            residual = np.linalg.norm(y - y_hat)  # L2 norm (reconstruction error)

            anomaly_map[i, j] = residual

    return anomaly_map

def crd_rgb_detection_robust(img_rgb, win_out=9, win_in=3, lamda=1e-1):
    img_data = img_rgb.astype(np.float32)
    h, w, bands = img_data.shape
    r_out, r_in = win_out // 2, win_in // 2
    
    anomaly_map = np.zeros((h, w))
    pad_img = np.pad(img_data, ((r_out, r_out), (r_out, r_out), (0, 0)), mode='edge')
    
    mask = np.ones((win_out, win_out), dtype=bool)
    mask[r_out-r_in : r_out+r_in+1, r_out-r_in : r_out+r_in+1] = False
    
    for i in range(h):
        for j in range(w):
            y = img_data[i, j, :].reshape(-1, 1)
            window = pad_img[i : i+win_out, j : j+win_out, :]
            D = window[mask].T  # shape: (3, n)
            
            # --- 修正ポイント：行列演算の安定化 ---
            DtD = np.dot(D.T, D)
            n_pixels = D.shape[1]
            
            # 正則化項を追加
            A = DtD + lamda * np.eye(n_pixels)
            b = np.dot(D.T, y)
            
            # linalg.solve の代わりに lstsq (最小二乗法) を使用
            # これにより、行列が完全に特異でもエラーにならずに近似解を返します
            alpha, _, _, _ = linalg.lstsq(A, b)
            
            y_hat = np.dot(D, alpha)
            anomaly_map[i, j] = np.linalg.norm(y - y_hat)
                
    return anomaly_map

# 1. 画像の読み込み
# 任意の画像ファイルを指定してください（例: 'sample.jpg'）
# ここでは動作確認用にダミー画像を生成します
img_bgr = cv2.imread('/content/001.png') 

if img_bgr is None:
    # 画像がない場合のサンプルデータ生成（100x100の緑背景に赤い点）
    img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    img_rgb[:, :, 1] = 150  # 背景を緑に
    img_rgb[50:53, 50:53, 0] = 255  # 異常（赤い点）
    img_rgb = cv2.GaussianBlur(img_rgb, (3,3), 0)
else:
    # OpenCVはBGRなのでRGBに変換
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. CRD実行
# win_out（外窓）は背景のテクスチャサイズに合わせて調整
# win_in（ガード窓）は検出したい異常のサイズより少し大きく設定
print("Processing CRD... please wait.")
score_map = crd_rgb_detection_robust(img_rgb, win_out=9, win_in=5, lamda=0.1)

# 3. 結果の表示
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original RGB")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("CRD Anomaly Score")
plt.imshow(score_map, cmap='jet')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Binary Detection")
# 上位1%を異常として表示
thresh = np.percentile(score_map, 99)
plt.imshow(score_map > thresh, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
