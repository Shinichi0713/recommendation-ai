import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import coins # サンプル画像としてコイン画像を使用
from skimage.transform import resize
from sklearn.feature_extraction.image import extract_patches_2d

# GPUが利用可能かチェック
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Deep Unfolding モデル (LISTAベースのRPCA)
class DeepUnfoldingRPCA_Image(nn.Module):
    def __init__(self, patch_dim, layers=10):
        super(DeepUnfoldingRPCA_Image, self).__init__()
        self.layers = layers
        # 学習可能な閾値（各層で異なる値を学習）
        # 初期値を小さめに設定し、積極的にスパース成分を分離する
        self.thetas = nn.Parameter(torch.ones(layers) * 0.01)
        # 重み行列 (入力次元 x 入力次元)。ここでは恒等行列に近く初期化
        self.W = nn.Parameter(torch.eye(patch_dim).to(device))

    def soft_threshold(self, x, theta):
        # 軟しきい値処理: 異常(Sparse成分)を抽出する核
        return torch.sign(x) * torch.relu(torch.abs(x) - theta)

    def forward(self, M):
        # M: 入力画像パッチ（Flattenされていると想定）
        # S: 推定される異常成分 (Sparse)
        S = torch.zeros_like(M).to(device)
        
        for i in range(self.layers):
            # RPCAの反復ステップを展開
            # residual: 入力Mから現在の異常推定Sを引いた「正常と仮定される部分」
            residual = M - S 
            
            # Sの更新: residualに対してWを適用し、しきい値処理でスパース性を強制
            S = self.soft_threshold(S + torch.matmul(residual, self.W), self.thetas[i])
            
        # L: 低ランク成分 (正常な部分) = 入力 - 異常
        L = M - S 
        return L, S

# 2. データ準備
# 画像の読み込みと前処理
image = coins()[160:210, 50:100] # 一部を切り出してシンプルに
image = resize(image, (64, 64), anti_aliasing=True)
image = image / np.max(image) # 0-1に正規化

# 画像を表示して確認
plt.imshow(image, cmap='gray')
plt.title("Original Image (Sample)")
plt.axis('off')
plt.show()

# 画像パッチの抽出
patch_size = (8, 8)
n_patches_per_image = 100 # 1枚の画像から抽出するパッチ数
patches = extract_patches_2d(image, patch_size)
# パッチをランダムに選択（学習データ量を調整）
np.random.shuffle(patches)
patches = patches[:n_patches_per_image]

# パッチをフラット化し、PyTorchのテンソルに変換
patch_dim = patch_size[0] * patch_size[1] # 例: 8x8 = 64
data_patches = torch.tensor(patches.reshape(-1, patch_dim), dtype=torch.float32).to(device)

print(f"Number of patches: {data_patches.shape[0]}")
print(f"Patch dimension: {data_patches.shape[1]}")

# 3. 学習の設定
model = DeepUnfoldingRPCA_Image(patch_dim).to(device)
# オプティマイザ
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 損失関数:
# 1. 再構成誤差: L + S が M にどれだけ近いか (MSE)
# 2. スパース性罰則: S がどれだけスパースか (L1 norm)
lambda_sparse = 0.01 # スパース性を強制する重み

# 4. 学習ループ (教師なし学習)
print("Training Deep Unfolding for Image Anomaly Detection...")
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # 順伝播
    L_pred, S_pred = model(data_patches)
    
    # 損失計算
    reconstruction_loss = F.mse_loss(data_patches, L_pred + S_pred)
    sparse_penalty = torch.mean(torch.norm(S_pred, p=1, dim=1)) # 各パッチのSのL1ノルム平均
    
    loss = reconstruction_loss + lambda_sparse * sparse_penalty
    
    # 逆伝播と最適化
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}, Rec Loss: {reconstruction_loss.item():.6f}, Sparse Penalty: {sparse_penalty.item():.6f}")

print("Training finished.")

# 5. 結果の可視化 (学習済みのパッチからLとSを抽出)
model.eval()
with torch.no_grad():
    sample_patch_idx = np.random.randint(0, data_patches.shape[0])
    test_patch = data_patches[sample_patch_idx].unsqueeze(0)
    
    L_extracted, S_extracted = model(test_patch)
    
    # 元の画像形式に戻す
    original_patch_img = test_patch.cpu().numpy().reshape(patch_size)
    L_img = L_extracted.cpu().numpy().reshape(patch_size)
    S_img = S_extracted.cpu().numpy().reshape(patch_size)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_patch_img, cmap='gray')
plt.title("Original Patch")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(L_img, cmap='gray')
plt.title("Low-Rank (Normal)")
plt.axis('off')

plt.subplot(1, 3, 3)
# 異常部分を強調するために、S_imgの値を正規化またはクリップして表示
# S_imgはほとんど0なので、表示を工夫
display_S_img = np.clip(S_img, 0, np.max(S_img) * 0.5) if np.max(S_img) > 0 else S_img
plt.imshow(display_S_img, cmap='hot') # 異常部分が赤く表示されるように
plt.title("Sparse (Anomaly)")
plt.axis('off')

plt.tight_layout()
plt.show()