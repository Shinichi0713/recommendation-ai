import torch
import matplotlib.pyplot as plt
import numpy as np

# ------------------
# 1 sample 取得
# ------------------
model.eval()

with torch.no_grad():
    data = next(iter(train_loader))
    # labelsがある場合も考慮して、最初の要素(images)を取得
    inputs = data[0].to(device)   # [Batch, C, H, W]

    low_rank, sparse = model(inputs)
    recon = low_rank + sparse

# ------------------
# Tensor → numpy (修正版)
# ------------------
def tensor_to_img(x):
    """
    x: [Batch, C, H, W] から先頭の1枚を抽出して変換
    """
    # バッチの先頭 [0] を指定することで、確実に3次元 [C, H, W] にする
    x = x[0].detach().cpu()        
    x = torch.clamp(x, 0, 1)
    
    # グレースケール(C=1)の場合、permuteせずに [H, W] にする
    if x.shape[0] == 1:
        return x.squeeze(0).numpy() # [H, W]
    
    # カラー(C=3)の場合
    return x.permute(1, 2, 0).numpy()  # [H, W, C]

# 各画像を変換
input_img = tensor_to_img(inputs)
low_rank_img = tensor_to_img(low_rank)
recon_img = tensor_to_img(recon)

# ------------------
# Anomaly map (修正版)
# ------------------
# バッチの先頭 [0] の sparse 成分からアノマリーマップを作成
sparse_sample = sparse[0] # [C, H, W]
anomaly_map = torch.mean(torch.abs(sparse_sample), dim=0) # Channel方向に平均 -> [H, W]
anomaly_map = anomaly_map.cpu().numpy()

# 正規化（0〜1）
if anomaly_map.max() > anomaly_map.min():
    anomaly_map = (anomaly_map - anomaly_map.min()) / \
                  (anomaly_map.max() - anomaly_map.min() + 1e-8)

# ------------------
# 表示
# ------------------
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(input_img, cmap='gray'); plt.title("Input")
plt.subplot(1, 3, 2); plt.imshow(low_rank_img, cmap='gray'); plt.title("Low-Rank (Background)")
plt.subplot(1, 3, 3); plt.imshow(anomaly_map, cmap='hot'); plt.title("Anomaly Map")
plt.show()