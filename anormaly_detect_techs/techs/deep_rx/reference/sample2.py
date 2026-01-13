import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# 1. 特徴抽出器 (ResNet18) の定義
class DeepFeatureExtractor(nn.Module):
    def __init__(self):
        super(DeepFeatureExtractor, self).__init__()
        # 訓練済みモデルを使用。layer3までの特徴を使う（解像度と意味のバランスが良い）
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-3]) 
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.features(x) # 出力サイズ: [Batch, 256, 14, 14] (224px入力時)

# 2. Deep RX エンジン
class DeepRXDetector:
    def __init__(self):
        self.mu = None
        self.sigma_inv = None

    def fit(self, feature_list):
        # 全画像から集めた特徴ベクトル [N, 256] を結合
        all_features = np.concatenate(feature_list, axis=0)
        print(f"学習データ数（パッチ単位）: {all_features.shape[0]}")
        
        # 平均と共分散行列（およびその逆行列）を計算
        self.mu = np.mean(all_features, axis=0)
        sigma = np.cov(all_features, rowvar=False)
        # 数値的安定化のために微小な値を加算（正則化）
        self.sigma_inv = np.linalg.inv(sigma + 0.01 * np.eye(sigma.shape[0]))
        print("学習完了: 正常データの統計量を保存しました。")

    def score(self, features):
        # マハラノビス距離（RXスコア）の計算
        # features: [H*W, 256]
        diff = features - self.mu
        # ベクトル化した一括計算
        scores = np.sum(np.dot(diff, self.sigma_inv) * diff, axis=1)
        return scores

# --- 実行プロセス ---

# 設定
TRAIN_DIR = './train'  # 正常画像のみが入ったフォルダ
TEST_IMAGE = 'anomaly_sample.jpg' # テストしたい画像
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = DeepFeatureExtractor()
detector = DeepRXDetector()

# 3. 学習フェーズ (正常画像の読み込み)
print(f"'{TRAIN_DIR}' から正常データを読み込んでいます...")
train_features = []
image_files = glob.glob(os.path.join(TRAIN_DIR, "*.*")) # jpg, pngなど

for fpath in image_files:
    try:
        img = Image.open(fpath).convert('RGB')
        img_t = transform(img).unsqueeze(0)
        feat = model(img_t) # [1, 256, 14, 14]
        # [14*14, 256] に変換してリストに追加
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, 256).numpy()
        train_features.append(feat_flat)
    except Exception as e:
        print(f"スキップ: {fpath} ({e})")

detector.fit(train_features)

# 4. 推論フェーズ (異常検知)
if os.path.exists(TEST_IMAGE):
    test_img_raw = Image.open(TEST_IMAGE).convert('RGB')
    test_img_t = transform(test_img_raw).unsqueeze(0)
    
    test_feat = model(test_img_t)
    _, c, h, w = test_feat.shape
    test_feat_flat = test_feat.permute(0, 2, 3, 1).reshape(-1, c).numpy()
    
    # RXスコア算出
    raw_scores = detector.score(test_feat_flat)
    score_map = raw_scores.reshape(h, w)
    
    # 5. 可視化
    # ヒートマップを元の画像サイズに拡大
    score_map_img = Image.fromarray(score_map).resize(test_img_raw.size, resample=Image.BILINEAR)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_img_raw); plt.title("Input Image")
    plt.subplot(1, 2, 2)
    plt.imshow(score_map_img, cmap='jet'); plt.title("Deep RX Anomaly Score")
    plt.colorbar(); plt.show()
else:
    print(f"テスト画像 {TEST_IMAGE} が見つかりません。")