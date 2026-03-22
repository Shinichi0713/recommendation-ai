import numpy as np

# 1. ユーザー評価データの定義 (アクション, 恋愛, SF, ホラー)
users = {
    "User_A": np.array([5, 1, 5, 1]), # アクションとSFが大好き
    "User_B": np.array([1, 5, 2, 4]), # 恋愛とホラーが好き
    "User_C": np.array([4, 2, 4, 2]), # User_Aに近い好み
    "User_D": np.array([3, 3, 3, 3])  # 全ジャンル平均的
}

def cosine_similarity(v1, v2):
    # 内積 (a, b) = a1*b1 + ... + an*bn
    dot_product = np.dot(v1, v2)
    
    # ノルム (長さ) ||v|| = sqrt((v, v))
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # コサイン類似度の計算
    return dot_product / (norm_v1 * norm_v2)

# User_A と各ユーザーの類似度を計算
target = "User_A"
print(f"--- {target} との類似度計算 ---")

for name, vec in users.items():
    if name == target: continue
    
    sim = cosine_similarity(users[target], vec)
    print(f"{name:6}: {sim:.4f}")

# 結果の解釈
# 値が 1 に近いほど「内積空間で同じ方向を向いている（好みが似ている）」ことを示します。

