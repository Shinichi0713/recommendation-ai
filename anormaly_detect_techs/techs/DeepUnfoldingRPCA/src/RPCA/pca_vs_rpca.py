import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score

# --- 1. RPCA (IALM) の定義 ---
def svt(M, tau):
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    S_thresh = np.maximum(S - tau, 0)
    return np.dot(U * S_thresh, Vh)

def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def ialm_rpca(M, max_iter=30):
    m, n = M.shape
    lam = 1.0 / np.sqrt(max(m, n))
    mu = 1.25 / np.linalg.norm(M, ord=2)
    rho = 1.5
    L, S, Y = np.zeros_like(M), np.zeros_like(M), np.zeros_like(M)
    for i in range(max_iter):
        L = svt(M - S + (1/mu) * Y, 1/mu)
        S = soft_threshold(M - L + (1/mu) * Y, lam/mu)
        Y = Y + mu * (M - L - S)
        mu *= rho
    return L, S

# --- 2. パッチ処理関数 ---
def image_to_patches(img, patch_size):
    h, w = img.shape
    patches = []
    # 画像サイズがパッチサイズで割り切れる範囲のみ処理
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patches.append(img[i:i+patch_size, j:j+patch_size].flatten())
    return np.array(patches).T

def patches_to_image(patches, img_shape, patch_size):
    h, w = img_shape
    # 再構成後のサイズを計算
    h_new = (h // patch_size) * patch_size
    w_new = (w // patch_size) * patch_size
    img = np.zeros((h_new, w_new))
    idx = 0
    for i in range(0, h_new, patch_size):
        for j in range(0, w_new, patch_size):
            img[i:i+patch_size, j:j+patch_size] = patches[:, idx].reshape(patch_size, patch_size)
            idx += 1
    return img

# --- 3. メイン処理 ---
# ファイルパス
img_path = '/content/exp1_num_3667.jpg'
mask_path = '/content/exp1_num_3667.png'

# 画像読み込み (ダミー生成コードは削除しました)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
# マスクが0/1画像の場合、0より大きい場所をTrueとする
gt_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
gt_mask = gt_mask_raw > 0 

if img is None or gt_mask_raw is None:
    raise ValueError("画像の読み込みに失敗しました。パスを確認してください。")

patch_size = 8 * 4
M = image_to_patches(img, patch_size)

# PCA による検知 (同じインスタンスを使用)
pca_model = PCA(n_components=1)
low_rank_pca = pca_model.fit_transform(M.T)
L_pca_flat = pca_model.inverse_transform(low_rank_pca).T
S_pca_flat = np.abs(M - L_pca_flat)
S_pca_img = patches_to_image(S_pca_flat, img.shape, patch_size)

# RPCA による検知
L_rpca_flat, S_rpca_flat = ialm_rpca(M)
S_rpca_img = patches_to_image(np.abs(S_rpca_flat), img.shape, patch_size)

# --- サイズ調整 ---
h_res, w_res = S_pca_img.shape
gt_mask_cropped = gt_mask[:h_res, :w_res]

# 二値化マスクの作成 (閾値: 上位2%を異常とする)
thresh_pca = np.percentile(S_pca_img, 98)
mask_pca = S_pca_img > thresh_pca
thresh_rpca = np.percentile(S_rpca_img, 98)
mask_rpca = S_rpca_img > thresh_rpca

# --- 4. 評価 ---
def evaluate(gt, pred):
    p = precision_score(gt.flatten(), pred.flatten(), zero_division=0)
    r = recall_score(gt.flatten(), pred.flatten(), zero_division=0)
    f1 = f1_score(gt.flatten(), pred.flatten(), zero_division=0)
    return p, r, f1

p_pca, r_pca, f1_pca = evaluate(gt_mask_cropped, mask_pca)
p_rpca, r_rpca, f1_rpca = evaluate(gt_mask_cropped, mask_rpca)

# --- 5. 可視化 ---
titles = ["Original (Cropped)", "GT Mask", "PCA Result", "RPCA Result"]
imgs = [img[:h_res, :w_res], gt_mask_cropped, mask_pca, mask_rpca]

plt.figure(figsize=(16, 4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.show()

print(f"[PCA]  Precision: {p_pca:.3f}, Recall: {r_pca:.3f}, F1: {f1_pca:.3f}")
print(f"[RPCA] Precision: {p_rpca:.3f}, Recall: {r_rpca:.3f}, F1: {f1_rpca:.3f}")