import numpy as np
import matplotlib.pyplot as plt

class CRDDetector:
    def __init__(self, hsi_cube):
        """
        Args:
            hsi_cube (np.ndarray): (H, W, Bands)
        """
        self.data = hsi_cube.astype(np.float32)
        self.h, self.w, self.bands = hsi_cube.shape

    def detect(self, win_inner=3, win_outer=7, lamda=1e-6):
        """
        CRDアルゴリズムによる異常検知
        Args:
            win_inner (int): 内側の窓サイズ（ターゲットを除外するガード用）
            win_outer (int): 外側の窓サイズ（背景辞書を作成する範囲）
            lamda (float): 正則化パラメータ（計算の安定化）
        """
        half_out = win_outer // 2
        half_in = win_inner // 2
        
        # スコアマップの初期化
        score_map = np.zeros((self.h, self.w))
        
        # 境界でのエラーを避けるためパディング
        padded_data = np.pad(self.data, ((half_out, half_out), (half_out, half_out), (0, 0)), mode='edge')
        
        # 単位行列（正則化用）
        I = np.eye(self.bands) # 実際には背景画素数に基づくが、カーネル化の実装ではBandsに依存

        print(f"Running CRD Detection (Window: {win_inner}x{win_inner} to {win_outer}x{win_outer})...")
        
        # 各画素をスキャン
        for y in range(self.h):
            for x in range(self.w):
                # パディング後の座標
                py, px = y + half_out, x + half_out
                
                # ターゲット画素 y
                target_pixel = padded_data[py, px, :].reshape(-1, 1)
                
                # 背景辞書 X の作成（二重窓の間の画素を抽出）
                outer_region = padded_data[py-half_out : py+half_out+1, px-half_out : px+half_out+1, :]
                
                # 内側の窓を除外して背景画素をフラットにする
                mask = np.ones((win_outer, win_outer), dtype=bool)
                start_in = half_out - half_in
                end_in = start_in + win_inner
                mask[start_in:end_in, start_in:end_in] = False
                
                # 背景画素行列 X (Bands, Number of background pixels)
                X = outer_region[mask].T 
                
                # --- CRDのコア計算 ---
                # w = (X^T * X + lambda * I)^-1 * X^T * y
                # ただし、画素数が多い場合は Woodburyの恒等式的なアプローチや
                # ターゲット y に対する残差を直接計算する
                
                # 重み w の計算
                # コラボレーティブ表現: target ≈ X * w
                # 閉形式解: w = inv(X.T @ X + lambda * I) @ X.T @ target
                X_t = X.T
                W = np.linalg.solve(X_t @ X + lamda * np.eye(X.shape[1]), X_t @ target_pixel)
                
                # 残差（異常スコア）: ||y - Xw||_2
                residual = np.linalg.norm(target_pixel - X @ W)
                score_map[y, x] = residual
                
        return score_map

# --- 実行・可視化コード ---
# detector = CRDDetector(filtered_cube) # フィルタリング後のデータを入れるのが推奨
# crd_scores = detector.detect(win_inner=3, win_outer=9, lamda=0.01)

# plt.imshow(crd_scores, cmap='jet')
# plt.colorbar(label='Anomaly Score')
# plt.title('CRD Detection Result')
# plt.show()