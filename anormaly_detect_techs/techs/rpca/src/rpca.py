import numpy as np
import matplotlib.pyplot as plt

class RPCADetector:
    def __init__(self, hsi_cube):
        """
        Args:
            hsi_cube (np.ndarray): (H, W, Bands)
        """
        self.h, self.w, self.bands = hsi_cube.shape
        # 3次元キューブを2次元行列 (Bands, Pixels) に変換
        # 行：波長、列：各画素 とすることで、全画素に共通するスペクトル構造を低ランク成分とする
        self.D = hsi_cube.reshape(-1, self.bands).T 
        
    def _soft_threshold(self, x, tau):
        """ソフトしきい値演算 (L1ノルムの近接作用素)"""
        return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

    def _svd_threshold(self, x, tau):
        """特異値しきい値演算 (核ノルムの近接作用素)"""
        U, S, Vh = np.linalg.svd(x, full_matrices=False)
        S_thresh = self._soft_threshold(S, tau)
        return (U * S_thresh) @ Vh

    def decompose(self, lamda=None, max_iter=100, tol=1e-7):
        """
        ADMMアルゴリズムによるロバスト主成分分析
        """
        n1, n2 = self.D.shape
        if lamda is None:
            lamda = 1 / np.sqrt(max(n1, n2))
            
        # 初期化
        L = np.zeros_like(self.D)
        S = np.zeros_like(self.D)
        Y = np.zeros_like(self.D) # ラグランジュ乗数
        mu = (n1 * n2) / (4.0 * np.linalg.norm(self.D, ord=1))
        
        print(f"Running RPCA Decomposition (lambda={lamda:.4f})...")
        
        for i in range(max_iter):
            # 1. L (低ランク成分) の更新
            L = self._svd_threshold(self.D - S + (1/mu) * Y, 1/mu)
            
            # 2. S (スパース成分) の更新
            S = self._soft_threshold(self.D - L + (1/mu) * Y, lamda/mu)
            
            # 3. Y (乗数) の更新と収束判定
            Z = self.D - L - S
            Y = Y + mu * Z
            
            err = np.linalg.norm(Z, 'fro') / np.linalg.norm(self.D, 'fro')
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}: error = {err:.2e}")
            
            if err < tol:
                break
                
        # 結果を元の3次元形状に復元
        # スコアマップは各画素のスパース成分のエネルギー（L2ノルム）として算出
        sparse_cube = S.T.reshape(self.h, self.w, self.bands)
        low_rank_cube = L.T.reshape(self.h, self.w, self.bands)
        
        # 異常スコア：全バンドにおけるスパース成分の強度の合計
        anomaly_score = np.linalg.norm(sparse_cube, axis=2)
        
        return low_rank_cube, sparse_cube, anomaly_score

# --- 使用例 ---
# detector = RPCADetector(hsi_cube)
# L_cube, S_cube, scores = detector.decompose(max_iter=50)

# # 可視化
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].imshow(detector.get_pseudo_rgb(hsi_cube)) # 以前のメソッド等を利用
# axes[0].set_title("Original RGB")
# im = axes[1].imshow(scores, cmap='hot')
# axes[1].set_title("RPCA Anomaly Score")
# plt.colorbar(im, ax=axes[1])
# plt.show()