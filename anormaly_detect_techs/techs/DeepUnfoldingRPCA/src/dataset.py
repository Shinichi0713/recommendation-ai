import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random

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


class NEUSegDataset(Data.Dataset):
    def __init__(self, base_dir, mode='train', base_size=256, transform=None):
        self.img_dir = os.path.join(base_dir, mode, 'images')
        self.mask_dir = os.path.join(base_dir, mode, 'masks')
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.base_size = base_size
        self.mode = mode
        self.transform = transform

    def __getitem__(self, idx):
        name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # 1. 基本のリサイズ
        image = TF.resize(image, (self.base_size, self.base_size))
        mask = TF.resize(mask, (self.base_size, self.base_size), interpolation=transforms.InterpolationMode.NEAREST)

        # 2. 訓練時のデータ拡張（PIL Imageのまま実行）
        if self.mode == 'train':
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # 3. 外部Transformの適用
        # もし外部Transformに ToTensor() が含まれているならそのまま適用
        # 含まれていない場合を考慮し、最後にTensor化を確認する
        if self.transform:
            image = self.transform(image)
        
        # もしtransform適用後もPIL ImageのままならTensorにする
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)
            
        # マスクは常にここでTensor化
        mask = TF.to_tensor(mask)

        # 4. マスクをバイナリにする
        mask = (mask > 0.5).float()

        return image, mask

    def __len__(self):
        return len(self.img_names)