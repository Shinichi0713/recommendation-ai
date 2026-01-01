import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. Datasetの定義
class NEUSegDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("L") # グレースケール
        if self.transform:
            image = self.transform(image)
        return image

# 2. RPCANet モデルの実装 (Deep Unfolding RPCA)
class RPCANet(nn.Module):
    def __init__(self, layers=5):
        super(RPCANet, self).__init__()
        self.layers = layers
        # ISTAのステップサイズとしきい値を層ごとに学習
        self.eta = nn.Parameter(torch.ones(layers) * 0.1)
        self.theta = nn.Parameter(torch.ones(layers) * 0.01)
        
        # 変換行列（畳み込み層として実装することで近傍情報を活用）
        self.conv_W = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False) for _ in range(layers)
        ])
        for conv in self.conv_W:
            nn.init.constant_(conv.weight, 1.0/9.0) # 平均化フィルタに近い初期値

    def soft_threshold(self, x, theta):
        return torch.sign(x) * torch.relu(torch.abs(x) - F.softplus(theta))

    def forward(self, M):
        S = torch.zeros_like(M)
        for i in range(self.layers):
            # L = M - S (低ランク成分の推定)
            # 勾配降下ステップの展開
            residual = M - S
            grad = self.conv_W[i](residual)
            S = self.soft_threshold(S + self.eta[i] * grad, self.theta[i])
            
        L = M - S
        return L, S

# 3. 学習の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "./datasets/NEU_Seg_Custom/train/images" # 先ほど整理したパス

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = NEUSegDataset(dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = RPCANet(layers=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 学習ループ
print("Training RPCANet...")
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for images in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        
        L, S = model(images)
        
        # 損失関数: 再構成誤差 + スパース性制約(L1)
        # M = L + S であることを保証しつつ、Sをまばらにする
        loss = F.mse_loss(L + S, images) + 0.1 * torch.mean(torch.abs(S))
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader):.6f}")
