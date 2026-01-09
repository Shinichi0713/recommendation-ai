from scipy.linalg import hankel
import numpy as np

# 元の時系列データ
data = np.array([1, 2, 3, 4, 5, 6, 7])

# 第1列と最後の行を指定して作成
# 窓幅3の場合
c = data[:3]  # [1, 2, 3]
r = data[2:]  # [3, 4, 5, 6, 7]
H = hankel(c, r)

print(H)
# 出力:
# [[1 2 3 4 5]
#  [2 3 4 5 6]
#  [3 4 5 6 7]]