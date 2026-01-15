import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as TF

# 1. 回転予測用モデルの定義
class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        # 特徴抽出器としてResNet18を使用
        self.backbone = models.resnet18(pretrained=True)
        # 最終層を4クラス分類（0, 90, 180, 270度）に変更
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 4)

    def forward(self, x):
        return self.backbone(x)

# 2. 学習ループ（正常データのみを使用）
def train_rotation_net(model, dataloader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        for inputs, _ in dataloader: # ラベルは無視
            # 1つの画像から4つの回転バリエーションを作成
            rotated_images = []
            labels = []
            for angle_idx, angle in enumerate([0, 90, 180, 270]):
                rotated_img = TF.rotate(inputs, angle)
                rotated_images.append(rotated_img)
                labels.append(torch.full((inputs.size(0),), angle_idx))
            
            # まとめてバッチ化
            inputs_combined = torch.cat(rotated_images)
            labels_combined = torch.cat(labels).to(device)
            inputs_combined = inputs_combined.to(device)

            optimizer.zero_grad()
            outputs = model(inputs_combined)
            loss = criterion(outputs, labels_combined)
            loss.backward()
            optimizer.step()

# 3. 異常判定（推論）
def detect_anomaly(model, image):
    model.eval()
    with torch.no_grad():
        # テスト画像を4回回転させて入力し、それぞれのSoftmax確率を取得
        probs = []
        for angle in [0, 90, 180, 270]:
            img_t = TF.rotate(image, angle).unsqueeze(0).to(device)
            output = model(img_t)
            prob = torch.softmax(output, dim=1)
            # 正解の回転角度に対する確率を取り出す
            probs.append(prob[0, angle//90].item())
        
        # 4つの予測の平均確信度を計算。低いほど「異常」
        anomaly_score = 1.0 - np.mean(probs)
    return anomaly_score

