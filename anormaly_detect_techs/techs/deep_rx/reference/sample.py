import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features
    

class DeepRX:
    def __init__(self):
        self.mu = None
        self.sigma_inv = None

    def train(self, feature_list):
        features = np.concatenate(feature_list, axis=0) # [N, 512]
        self.mu = np.mean(features, axis=0)
        # 共分散行列の計算と安定化
        sigma = np.cov(features, rowvar=False) + 0.01 * np.eye(features.shape[1])
        self.sigma_inv = np.linalg.inv(sigma)

    def predict(self, feature):
        diff = feature - self.mu
        score = np.dot(np.dot(diff, self.sigma_inv), diff.T)
        return score
    


