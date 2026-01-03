
from skimage.util import view_as_windows
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

patch_size = 16
stride = 8

dir_current = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_current)

# --- 画像読み込み（グレースケール推奨） ---
img = cv2.imread("gray_mountain.png", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0

# --- 前提条件（質問のコード部分） ---
# patch_size, stride, img は定義済みとします
patches = view_as_windows(img, (patch_size, patch_size), step=stride)
original_shape = patches.shape  # (h_nodes, w_nodes, patch_h, patch_w)
patches_flattened = patches.reshape(-1, patch_size * patch_size)

U, S, Vt = np.linalg.svd(patches_flattened, full_matrices=False)

# --- 背景画像の可視化手順 ---

# 1. 低ランク近似 (例: 上位 k 個の特異値のみを使用)
# 全ての成分を使うと元の画像に戻るため、kを小さく設定して「背景」を抽出します
k = 1 
low_rank_patches = (U[:, :k] * S[:k]) @ Vt[:k, :]

# 2. パッチの形状を元に戻す (h_nodes, w_nodes, patch_h, patch_w)
background_patches = low_rank_patches.reshape(original_shape)

# 3. パッチを結合して1枚の画像にする
# 重なり（stride）がある場合、単純な結合ではなく平均化が必要ですが、
# まずは確認用に一番左上の要素を並べる簡易的な復元を行います
h_nodes, w_nodes = original_shape[:2]
bg_img = np.zeros(((h_nodes-1)*stride + patch_size, (w_nodes-1)*stride + patch_size))

for i in range(h_nodes):
    for j in range(w_nodes):
        bg_img[i*stride : i*stride+patch_size, j*stride : j*stride+patch_size] = background_patches[i, j]

# 4. 表示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Background (Low-Rank, k={k})")
plt.imshow(bg_img, cmap='gray')
plt.show()