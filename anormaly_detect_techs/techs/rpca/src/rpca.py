import numpy as np
import matplotlib.pyplot as plt

class RPCADetector:
    def __init__(self, hsi_cube):
        self.h, self.w, self.bands = hsi_cube.shape
        # 行：波長 (Bands), 列：各画素 (H*W)
        self.D = hsi_cube.reshape(-1, self.bands).T 
        
    def get_pseudo_rgb(self, cube, rgb_bands=[40, 20, 10]):
        """データキューブから表示用の擬似RGBを作成する"""
        rgb_img = cube[:, :, rgb_bands].astype(np.float32)
        # チャンネルごとに正規化
        for i in range(3):
            c_min, c_max = rgb_img[:,:,i].min(), rgb_img[:,:,i].max()
            if c_max != c_min:
                rgb_img[:,:,i] = (rgb_img[:,:,i] - c_min) / (c_max - c_min)
        return rgb_img

    def _soft_threshold(self, x, tau):
        return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

    def _svd_threshold(self, x, tau):
        U, S, Vh = np.linalg.svd(x, full_matrices=False)
        S_thresh = self._soft_threshold(S, tau)
        return (U * np.diag(S_thresh)) @ Vh

    def decompose(self, lamda=None, max_iter=50, tol=1e-7):
        n1, n2 = self.D.shape
        if lamda is None:
            lamda = 1 / np.sqrt(max(n1, n2)) * 100
            
        L = np.zeros_like(self.D)
        S = np.zeros_like(self.D)
        Y = np.zeros_like(self.D)
        mu = (n1 * n2) / (4.0 * np.linalg.norm(self.D, ord=1))
        
        print(f"Running RPCA Decomposition...")
        
        for i in range(max_iter):
            # L (低ランク) と S (スパース) の更新
            L = self._svd_threshold(self.D - S + (1/mu) * Y, 1/mu)
            S = self._soft_threshold(self.D - L + (1/mu) * Y, lamda/mu)
            
            # 残差計算と収束判定
            Z = self.D - L - S
            Y = Y + mu * Z
            err = np.linalg.norm(Z, 'fro') / np.linalg.norm(self.D, 'fro')
            
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}: error = {err:.2e}")
            if err < tol: break
                
        # 3次元形状への復元
        low_rank_cube = L.T.reshape(self.h, self.w, self.bands)
        sparse_cube = S.T.reshape(self.h, self.w, self.bands)
        
        # 異常スコア：全バンドにおけるスパース成分のL2ノルム
        anomaly_score = np.linalg.norm(sparse_cube, axis=2)
        
        return low_rank_cube, sparse_cube, anomaly_score

# --- 実行セクション ---
# 1. インスタンス化
detector = RPCADetector(filtered_cube) 

# 2. 分解実行 (max_iterは計算時間短縮のため50程度に設定)
L_cube, S_cube, scores = detector.decompose(max_iter=50)

# 3. 可視化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 元の画像
axes[0].imshow(detector.get_pseudo_rgb(filtered_cube))
axes[0].set_title("Input HSI (Pseudo RGB)")
axes[0].axis('off')

# 低ランク成分（背景のみ）
axes[1].imshow(detector.get_pseudo_rgb(L_cube))
axes[1].set_title("Low-Rank Component (Background)")
axes[1].axis('off')

# 異常スコアマップ
im = axes[2].imshow(scores, cmap='hot')
axes[2].set_title("RPCA Anomaly Score (Sparse)")
axes[2].axis('off')
plt.colorbar(im, ax=axes[2], shrink=0.8)

plt.tight_layout()
plt.show()