import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. 特徴抽出モデルの定義 (Pre-trained ResNetを利用)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 学習済みのResNet18を使用
        resnet = models.resnet18(pretrained=True)
        # 最終層(fc)やpoolingを除き、途中の特徴マップを取り出す
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x) # [batch, 512, h/32, w/32]
        return features

# 2. Deep RX Detector クラス
class DeepRX:
    def __init__(self):
        self.mu = None
        self.sigma_inv = None

    def train(self, feature_list):
        """正常画像の特徴量から統計量を計算"""
        features = np.concatenate(feature_list, axis=0) # [N, 512]
        self.mu = np.mean(features, axis=0)
        # 共分散行列の計算と安定化
        sigma = np.cov(features, rowvar=False) + 0.01 * np.eye(features.shape[1])
        self.sigma_inv = np.linalg.inv(sigma)

    def predict(self, feature):
        """マハラノビス距離（RXスコア）を計算"""
        diff = feature - self.mu
        # Score = (x-mu)^T * Sigma^-1 * (x-mu)
        score = np.dot(np.dot(diff, self.sigma_inv), diff.T)
        return score

# --- メイン処理 ---

# 前処理の設定
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

extractor = FeatureExtractor()
detector = DeepRX()

# A. 学習 (正常画像数枚を使って背景統計を構築)
# 本来はループで複数の正常画像を読み込みます
normal_img = transform(Image.open('normal_sample.jpg').convert('RGB')).unsqueeze(0)
feat_map = extractor(normal_img) 
# 特徴マップの各位置をサンプルとして抽出 (空間情報を考慮)
feat_vectors = feat_map.permute(0, 2, 3, 1).reshape(-1, 512).numpy()
detector.train([feat_vectors])

# B. 推論 (異常を含む画像を読み込む)
test_img_raw = Image.open('anomaly_sample.jpg').convert('RGB')
test_img = transform(test_img_raw).unsqueeze(0)
test_feat_map = extractor(test_img)
h_feat, w_feat = test_feat_map.shape[2], test_feat_map.shape[3]

# 各画素位置（特徴マップ上）でスコア計算
test_feat_vectors = test_feat_map.permute(0, 2, 3, 1).reshape(-1, 512).numpy()
scores = np.array([detector.predict(v) for v in test_feat_vectors])

# C. 結果の可視化
score_map = scores.reshape(h_feat, w_feat)
# スコアマップを元の画像サイズにリサイズ
score_map_resized = Image.fromarray(score_map).resize(test_img_raw.size, resample=Image.BILINEAR)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.imshow(test_img_raw); plt.title("Original (Anomaly)")
plt.subplot(1, 2, 2); plt.imshow(score_map_resized, cmap='jet'); plt.title("Deep RX Score Map")
plt.colorbar(); plt.show()