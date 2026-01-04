import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. LISTA モデルの定義
class LISTA(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=5):
        super(LISTA, self).__init__()
        self.layers = layers
        # ISTAの数式: S = soft_threshold(S + W(M - DS))
        # LISTAではこれを簡略化した S = soft_threshold(We * M + S * S_matrix) 等の形式も使われますが
        # ここでは基本に忠実な展開を行います
        self.W = nn.Parameter(torch.randn(layers, hidden_dim, input_dim) * 0.1)
        self.S = nn.Parameter(torch.randn(layers, hidden_dim, hidden_dim) * 0.1)
        self.theta = nn.Parameter(torch.ones(layers, hidden_dim) * 0.01)

    def soft_threshold(self, x, theta):
        # 負の値を softplus で正に変換してからしきい値として使用
        t = F.softplus(theta)
        return torch.sign(x) * torch.relu(torch.abs(x) - t)

    def forward(self, M):
        # M: [batch, input_dim] (観測信号)
        # S: [batch, hidden_dim] (推定されるスパース信号)
        batch_size = M.size(0)
        S = torch.zeros(batch_size, self.S.size(-1)).to(M.device)
        
        for i in range(self.layers):
            # ISTAのステップを層として実行
            # S = soft_threshold(We * M + S_matrix * S, theta)
            z = F.linear(M, self.W[i]) + F.linear(S, self.S[i])
            S = self.soft_threshold(z, self.theta[i])
            
        return S

# 2. データの生成 (スパースな信号を合成)
input_dim = 50
hidden_dim = 100
n_samples = 1000

# 辞書行列 D (ランダム)
D = torch.randn(input_dim, hidden_dim)

# 真のスパース信号 S_true (10%だけ値がある)
S_true = (torch.randn(n_samples, hidden_dim) * (torch.rand(n_samples, hidden_dim) < 0.1).float())

# 観測信号 M = D * S_true
M = torch.matmul(S_true, D.t()) + torch.randn(n_samples, input_dim) * 0.01 # ノイズあり

# 3. 学習の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LISTA(input_dim, hidden_dim, layers=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

# 4. トレーニング
M, S_true = M.to(device), S_true.to(device)
print("Training LISTA...")
for epoch in range(200):
    optimizer.zero_grad()
    S_pred = model(M)
    loss = criterion(S_pred, S_true) # 教師あり学習：真のスパース信号に近づける
    loss.backward()
    optimizer.step()
    
    if epoch % 40 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 5. 結果の可視化
model.eval()
with torch.no_grad():
    sample_idx = 0
    S_pred_sample = model(M[sample_idx:sample_idx+1]).cpu().numpy().flatten()
    S_true_sample = S_true[sample_idx].cpu().numpy().flatten()

plt.figure(figsize=(12, 5))
plt.stem(S_true_sample, linefmt='g-', markerfmt='go', label='True Sparse Signal')
plt.stem(S_pred_sample, linefmt='r--', markerfmt='rx', label='LISTA Estimated')
plt.title("LISTA Signal Recovery")
plt.legend()
plt.show()