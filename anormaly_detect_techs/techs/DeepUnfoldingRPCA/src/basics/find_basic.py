import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

dir_current = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_current)

# --- 画像読み込み（グレースケール推奨） ---
img = cv2.imread("gray_mountain.png", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0

H, W = img.shape

# --- パッチ抽出 ---
patch_size = 16
stride = 8

patches = []
for y in range(0, H - patch_size + 1, stride):
    for x in range(0, W - patch_size + 1, stride):
        patch = img[y:y+patch_size, x:x+patch_size]
        patches.append(patch.flatten())

X = np.stack(patches)  # (num_patches, patch_dim)

# --- PCA ---
n_components = 8
pca = PCA(n_components=n_components)
pca.fit(X)

bases = pca.components_  # (n_components, patch_dim)

# --- 可視化 ---
plt.figure(figsize=(12, 3))
for i in range(n_components):
    plt.subplot(1, n_components, i+1)
    plt.imshow(bases[i].reshape(patch_size, patch_size), cmap="gray")
    plt.title(f"Basis {i+1}")
    plt.axis("off")
plt.show()
