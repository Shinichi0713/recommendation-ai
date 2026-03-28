import scipy.io
import numpy as np
import matplotlib.pyplot as plt


class HSIProcessor:
    def __init__(self, data_key='data'):
        """
        ハイパースペクトル画像を処理するためのクラス

        Args:
            file_path (str): .matファイルのパス
            data_key (str): .matファイル内のデータ本体に対応するキー名
        """
        self.data_key = data_key
        self.hsi_cube = None
        self.shape = None

    def load_data(self, file_path):
        """ファイルをロードし、データ本体を抽出する"""
        try:
            mat_data = scipy.io.loadmat(file_path)
            # 指定されたキーが存在するか確認
            if self.data_key not in mat_data:
                available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                raise KeyError(f"キー '{self.data_key}' が見つかりません。利用可能なキー: {available_keys}")

            self.hsi_cube = mat_data[self.data_key]
            self.shape = self.hsi_cube.shape
            print(f"Successfully loaded. Shape: {self.shape} (H, W, Bands)")
            return self.hsi_cube
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def get_pseudo_rgb(self, hsi_cube, rgb_bands=[40, 20, 10]):
        """
        指定したバンドを使って擬似RGB画像を作成する

        Args:
            rgb_bands (list): [R, G, B]に対応させるバンドインデックス

        Returns:
            np.ndarray: 正規化されたRGB画像 (H, W, 3)
        """
        if hsi_cube is None:
            raise ValueError("入力されたハイパースペクトル画像が無効です。")

        # 指定バンドの抽出
        rgb_img = hsi_cube[:, :, rgb_bands]

        # データの正規化 (0.0 - 1.0)
        img_min = rgb_img.min()
        img_max = rgb_img.max()
        if img_max - img_min != 0:
            rgb_img = (rgb_img - img_min) / (img_max - img_min)

        return rgb_img

    def show_rgb(self, hsi_cube, rgb_bands=[40, 20, 10], figsize=(8, 6)):
        """擬似RGB画像を表示する"""
        rgb_img = self.get_pseudo_rgb(hsi_cube, rgb_bands)

        plt.figure(figsize=figsize)
        plt.imshow(rgb_img)
        plt.title(f"Pseudo RGB (Bands: {rgb_bands[0]}, {rgb_bands[1]}, {rgb_bands[2]})")
        plt.axis('off')
        plt.show()


# --- 使用例 ---
path = '/content/abu-airport-1.mat'
processor = HSIProcessor(data_key='data')
cube_rgb = processor.load_data(path)
processor.show_rgb(cube_rgb, rgb_bands=[40, 20, 10])

import matplotlib.pyplot as plt

L = cube_rgb
height, width, band = L.shape

# L を SVD 分解
U, S, Vt = np.linalg.svd(L, full_matrices=False)

# 第kモードの空間パターンを表示 (画像サイズが height x width と仮定)
for k in range(40):
  mode_image = U[:, k].reshape(height, width)
  plt.imshow(mode_image, cmap='gray')
  plt.title(f"Rank-{k+1} Spatial Mode")
  plt.show()

# 第kモードの時間変化を表示
plt.plot(Vt[k, :])
plt.title(f"Rank-{k+1} Temporal Evolution")
plt.show()
