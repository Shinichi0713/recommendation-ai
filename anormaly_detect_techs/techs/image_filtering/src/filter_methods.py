import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from enum import Enum, auto

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

class FilterProcessor:
    def __init__(self):
        """
        Args:
            hsi_cube (np.ndarray): (H, W, Bands) の画像データ
        """
        pass

    def get_pseudo_rgb(self, hsi_cube, rgb_bands=[40, 20, 10]):
        rgb_img = hsi_cube[:, :, rgb_bands].astype(np.float32)
        img_min, img_max = rgb_img.min(), rgb_img.max()
        return (rgb_img - img_min) / (img_max - img_min)

    def _apply_frequency_filter(self, img, filter_type, radius=30):
        """周波数ドメインでのフィルタリング内部処理"""
        # 各チャンネル（R,G,B）ごとに処理
        filtered_channels = []
        for i in range(3):
            channel = img[:, :, i]
            # 1. FFT
            dft = np.fft.fft2(channel)
            dft_shift = np.fft.fftshift(dft)
            
            # 2. マスク作成
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.uint8)
            
            # 中心からの距離に基づいて円形マスクを作成
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
            
            if filter_type == FilterType.LOW_PASS:
                mask[mask_area] = 1
            else: # HIGH_PASS
                mask.fill(1)
                mask[mask_area] = 0
            
            # 3. マスク適用と逆FFT
            fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            filtered_channels.append(img_back)
            
        return np.stack(filtered_channels, axis=2)

    def apply_single_filter(self, img, filter_type, kernel_size=5):
        if isinstance(filter_type, int):
            filter_type = FilterType(filter_type)

        # OpenCVフィルタは通常 [0, 255] の uint8 を好むため変換が必要な場合があるが
        # ここでは float32 [0, 1] のまま処理（多くのOpenCV関数が対応）
        
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
            return cv2.bilateralFilter(img, d=9, sigmaColor=0.1, sigmaSpace=75)
        
        # --- 2. エッジ抽出 ---
        elif filter_type == FilterType.LAPLACIAN:
            # 2次微分
            lap = cv2.Laplacian(img, cv2.CV_32F)
            return np.clip(lap, 0, 1)
        elif filter_type == FilterType.SOBEL:
            # 水平・垂直の絶対値を合算
            sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=kernel_size)
            sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=kernel_size)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            return np.clip(sobel, 0, 1)
        elif filter_type == FilterType.CANNY:
            # Cannyはuint8 [0, 255] への変換が必須
            img_uint8 = (img * 255).astype(np.uint8)
            edges = []
            for i in range(3):
                edges.append(cv2.Canny(img_uint8[:,:,i], 100, 200))
            return np.stack(edges, axis=2).astype(np.float32) / 255.0

        # --- 3. 周波数ドメイン ---
        elif filter_type in [FilterType.LOW_PASS, FilterType.HIGH_PASS]:
            return self._apply_frequency_filter(img, filter_type)
        
        return img

    def show_filtered_results(self, hsi_cube, filter_list=None):
        if filter_list is None:
            filter_list = [f for f in FilterType]

        base_img = self.get_pseudo_rgb(hsi_cube)
        n = len(filter_list)
        cols = 3 # 1行に3枚表示
        rows = (n + cols - 1) // cols
        
        plt.figure(figsize=(15, 5 * rows))
        for i, f_type in enumerate(filter_list):
            filtered_img = self.apply_single_filter(base_img, f_type)
            title = FilterType(f_type).name if isinstance(f_type, int) else f_type.name
            
            plt.subplot(rows, cols, i + 1)
            plt.imshow(filtered_img, cmap='gray' if f_type == FilterType.CANNY else None)
            plt.title(title)
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