import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from enum import Enum

class FilterType(Enum):
    # 1. 平滑化
    ORIGINAL = 0
    AVERAGING = 1
    GAUSSIAN = 2
    MEDIAN = 3
    BILATERAL = 4
    # 2. エッジ抽出・鋭鋭化
    LAPLACIAN = 5
    SOBEL = 6
    CANNY = 7
    # 3. 周波数ドメイン
    LOW_PASS = 8
    HIGH_PASS = 9
    # 4. 高度なフィルタ
    GUIDED = 10
    NLM = 11        # Non-Local Means
    MORPH_OPEN = 12 # Morphology Opening

class FilterProcessor:
    def __init__(self):
        pass

    def get_pseudo_rgb(self, hsi_cube, rgb_bands=[40, 20, 10]):
        rgb_img = hsi_cube[:, :, rgb_bands].astype(np.float32)
        img_min, img_max = rgb_img.min(), rgb_img.max()
        return (rgb_img - img_min) / (img_max - img_min) if (img_max - img_min) != 0 else rgb_img

    def _apply_frequency_filter_2d(self, channel, filter_type, radius=30):
        """単一チャンネル（空間方向）に対する周波数フィルタリング"""
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
        
        mask = np.zeros((rows, cols), np.uint8)
        if filter_type == FilterType.LOW_PASS:
            mask[mask_area] = 1
        else: # HIGH_PASS
            mask.fill(1)
            mask[mask_area] = 0
            
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        return img_back

    def apply_single_filter(self, img, filter_type, kernel_size=5):
        """2D画像（単一バンドまたはRGB）に対してフィルタを適用"""
        if isinstance(filter_type, int):
            filter_type = FilterType(filter_type)
        
        # チャンネル数の確認
        is_single_channel = len(img.shape) == 2 or img.shape[2] == 1
        
        # --- 1. 平滑化 ---
        if filter_type == FilterType.ORIGINAL:
            return img
        elif filter_type == FilterType.AVERAGING:
            return cv2.blur(img, (kernel_size, kernel_size))
        elif filter_type == FilterType.GAUSSIAN:
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif filter_type == FilterType.MEDIAN:
            return cv2.medianBlur(img, kernel_size)
        elif filter_type == FilterType.BILATERAL:
            # 1枚ずつのバンドの場合はcvtColorが必要な場合があるため簡易化
            return cv2.bilateralFilter(img, d=9, sigmaColor=0.1, sigmaSpace=75)
        
        # --- 2. エッジ抽出 ---
        elif filter_type == FilterType.LAPLACIAN:
            return np.clip(cv2.Laplacian(img, cv2.CV_32F), 0, 1)
        elif filter_type == FilterType.SOBEL:
            sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=kernel_size)
            sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=kernel_size)
            return np.clip(np.sqrt(sx**2 + sy**2), 0, 1)
        elif filter_type == FilterType.CANNY:
            img_u8 = (img * 255).astype(np.uint8)
            if is_single_channel:
                return cv2.Canny(img_u8, 100, 200).astype(np.float32) / 255.0
            else:
                edges = [cv2.Canny(img_u8[:,:,i], 100, 200) for i in range(img.shape[2])]
                return np.stack(edges, axis=2).astype(np.float32) / 255.0

        # --- 3. 周波数ドメイン ---
        elif filter_type in [FilterType.LOW_PASS, FilterType.HIGH_PASS]:
            if is_single_channel:
                return self._apply_frequency_filter_2d(img, filter_type)
            else:
                res = [self._apply_frequency_filter_2d(img[:,:,i], filter_type) for i in range(img.shape[2])]
                return np.stack(res, axis=2)
        
        # --- 4. 高度なフィルタ ---
        if filter_type == FilterType.GUIDED:
            return cv2.ximgproc.guidedFilter(guide=img, src=img, radius=8, eps=0.01)
        elif filter_type == FilterType.NLM:
            img_u8 = (img * 255).astype(np.uint8)
            if is_single_channel:
                dst = cv2.fastNlMeansDenoising(img_u8, None, h=10, templateWindowSize=7, searchWindowSize=21)
            else:
                dst = cv2.fastNlMeansDenoisingColored(img_u8, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
            return dst.astype(np.float32) / 255.0
        elif filter_type == FilterType.MORPH_OPEN:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            img_u8 = (img * 255).astype(np.uint8)
            res = cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, kernel)
            return res.astype(np.float32) / 255.0
            
        return img

    def apply_hsi_filter(self, hsi_cube, filter_type, kernel_size=5):
        """
        HSIデータキューブの全バンドに対して空間方向にフィルタを一括適用
        Args:
            hsi_cube (np.ndarray): (H, W, Bands)
        Returns:
            filtered_hsi (np.ndarray): フィルタ適用後の (H, W, Bands)
        """
        h, w, bands = hsi_cube.shape
        filtered_hsi = np.zeros_like(hsi_cube, dtype=np.float32)
        
        # 処理を安定させるために [0, 1] に正規化（後で元に戻すか、正規化後のままにするか選択）
        hsi_min, hsi_max = hsi_cube.min(), hsi_cube.max()
        normalized_hsi = (hsi_cube - hsi_min) / (hsi_max - hsi_min) if hsi_max != hsi_min else hsi_cube
        
        print(f"Filtering {bands} bands using {FilterType(filter_type).name}...")
        for b in range(bands):
            band_img = normalized_hsi[:, :, b]
            # 空間方向にフィルタ適用
            filtered_hsi[:, :, b] = self.apply_single_filter(band_img, filter_type, kernel_size)
            
        return filtered_hsi

    def show_hsi_filter_comparison(self, hsi_cube, filter_list=[0, 2, 4, 10], rgb_bands=[40, 20, 10]):
        """空間フィルタ適用後のHSIを擬似RGBで比較表示"""
        n = len(filter_list)
        plt.figure(figsize=(5 * n, 5))
        
        for i, f_type in enumerate(filter_list):
            # HSI全体にフィルタ適用
            filtered_hsi = self.apply_hsi_filter(hsi_cube, f_type)
            # フィルタ後のHSIから擬似RGBを作成
            rgb_res = self.get_pseudo_rgb(filtered_hsi, rgb_bands)
            
            plt.subplot(1, n, i + 1)
            plt.imshow(rgb_res)
            title = FilterType(f_type).name if isinstance(f_type, int) else f_type.name
            plt.title(f"HSI + {title}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import read_hsi

    reader = read_hsi.HSIProcessor(file_path='data/sample_hsi.mat')
    cube = reader.load_data()
    if cube is not None:
        processor = FilterProcessor()
        processor.show_filtered_results(hsi_cube=cube, filter_list=[0, 5, 6, 7, 8, 9])
    else:
        print("Failed to load HSI data.")