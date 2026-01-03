import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 擬似的な多変量時系列データの生成
def generate_timeseries_data(n_samples=1000, anomaly=False):
    t = np.linspace(0, 50, n_samples)
    # 正常時：3つのセンサーが連動している（相関が高い状態）
    s1 = np.sin(t)
    s2 = np.sin(t + 0.5)
    s3 = s1 * 0.5 + s2 * 0.2 + np.random.normal(0, 0.05, n_samples)
    
    data = np.vstack([s1, s2, s3]).T
    
    if anomaly:
        # 700〜800サンプル目に異常を注入
        # センサー3だけが突然無関係な動き（スパイク）をする、または値が固定される
        data[700:800, 2] += 2.0  # スパイク異常
        data[850:900, 0] = 0.0   # 値の固着（故障）
        
    return pd.DataFrame(data, columns=['Sensor_A', 'Sensor_B', 'Sensor_C'])

# データ準備
train_df = generate_timeseries_data(n_samples=1000, anomaly=False) # 正常のみ
test_df = generate_timeseries_data(n_samples=1000, anomaly=True)   # 異常あり

# 2. 前処理（標準化が極めて重要）
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# 3. PCAモデルの構築
# 累積寄与率が90%以上になるように主成分数を決定（今回は3変数中2個に設定）
pca = PCA(n_components=2)
pca.fit(train_scaled)

# 4. 異常スコア（再構成誤差）の計算
def compute_anomaly_score(model, data_scaled):
    # データを主成分空間に射影し、再度元の空間に復元する
    projected = model.transform(data_scaled)
    reconstructed = model.inverse_transform(projected)
    
    # 元のデータと復元データの差（残差平方和）を計算
    mse = np.mean(np.power(data_scaled - reconstructed, 2), axis=1)
    return mse

train_scores = compute_anomaly_score(pca, train_scaled)
test_scores = compute_anomaly_score(pca, test_scaled)

# 閾値の設定（学習データの最大値や99パーセンタイルなど）
threshold = np.percentile(train_scores, 99)

# 5. 可視化
plt.figure(figsize=(15, 8))

# 元データのプロット
plt.subplot(2, 1, 1)
plt.plot(test_df.values)
plt.legend(test_df.columns)
plt.title("Multivariate Time Series (Test Data with Anomalies)")
plt.axvspan(700, 800, color='red', alpha=0.1, label="Anomaly Area")
plt.axvspan(850, 900, color='red', alpha=0.1)

# 異常スコアのプロット
plt.subplot(2, 1, 2)
plt.plot(test_scores, color='purple', label="Anomaly Score (MSE)")
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
plt.fill_between(range(len(test_scores)), 0, test_scores, where=(test_scores > threshold), color='red', alpha=0.3)
plt.title("Anomaly Detection via PCA Reconstruction Error")
plt.legend()

plt.tight_layout()
plt.show()

print(f"PCA Components: {pca.n_components_}")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")