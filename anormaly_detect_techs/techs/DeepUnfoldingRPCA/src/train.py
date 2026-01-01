import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Deep Unfolding モデル (ISTA-RPCA)
class DeepUnfoldingRPCA(nn.Module):
    def __init__(self, input_dim, layers=10):
        super(DeepUnfoldingRPCA, self).__init__()
        self.layers = layers
        # 学習可能なパラメータ：各層ごとの閾値（異常を拾う感度）
        # 初期値を小さめに設定し、確実に何かを拾う状態からスタートさせる
        self.thetas = nn.Parameter(torch.ones(layers) * 0.05)
        # 信号を変換する重み行列（恒等行列に近い状態からスタート）
        self.W = nn.Parameter(torch.eye(input_dim) * 0.5)

    def soft_threshold(self, x, theta):
        # 軟しきい値処理: これが異常(Sparse成分)を抽出する
        return torch.sign(x) * torch.relu(torch.abs(x) - theta)

    def forward(self, M):
        # S: 推定される異常成分 (Sparse)
        S = torch.zeros_like(M)
        
        for i in range(self.layers):
            # 行列分解の反復ステップを層として展開
            # 入力Mから現在の推定異常Sを引き、残った「正常っぽい部分」との差分を見る
            residual = M - S
            S = self.soft_threshold(S + torch.matmul(residual, self.W), self.thetas[i])
            
        return S

# 2. データの準備 (確実な検知のために異常を強調)
input_dim = 50
n_samples = 1000

# 正常データ: 滑らかなサイン波
t = torch.linspace(0, 5, input_dim)
normal_base = torch.sin(t).repeat(n_samples, 1)

# 異常データ: ランダムな場所に1つだけ巨大なスパイク（値=5.0）を置く
anomalies = torch.zeros(n_samples, input_dim)
for i in range(n_samples):
    idx = torch.randint(0, input_dim, (1,))
    anomalies[i, idx] = 5.0 # 非常に強い異常

input_data = normal_base + anomalies

# 3. 学習の設定
model = DeepUnfoldingRPCA(input_dim)
# 損失関数を「抽出したSが、実際の異常値と一致すること」に設定
criterion = nn.L1Loss() # 異常検知にはL1（絶対誤差）の方が鋭く反応する
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 学習ループ
print("Learning to detect anomalies...")
for epoch in range(500):
    optimizer.zero_grad()
    predicted_S = model(input_data)
    loss = criterion(predicted_S, anomalies)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 5. 可視化
model.eval()
with torch.no_grad():
    sample_idx = 0
    test_input = input_data[sample_idx].unsqueeze(0)
    detected_S = model(test_input)

plt.figure(figsize=(10, 5))
plt.plot(input_data[sample_idx].numpy(), label="Raw Input (Signal + Anomaly)", alpha=0.5)
plt.plot(anomalies[sample_idx].numpy(), label="True Anomaly (Target)", color="green", linestyle="--")
plt.plot(detected_S[0].numpy(), label="Detected Anomaly (Model Output)", color="red", linewidth=2)
plt.title("Deep Unfolding: Successful Anomaly Extraction")
plt.legend()
plt.grid(True)
plt.show()


