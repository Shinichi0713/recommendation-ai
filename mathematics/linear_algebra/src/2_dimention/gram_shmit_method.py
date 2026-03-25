import numpy as np

def gram_schmidt(vectors):
    """
    入力されたベクトルのリストから正規直交基底を生成する
    """
    basis = []
    for v in vectors:
        # 1. 既にある基底成分（影）をすべて差し引く（直交化）
        w = v.copy().astype(float)
        for b in basis:
            # プロジェクション（投影成分）を計算して引く
            w -= np.dot(v, b) * b
        
        # 2. 残った垂直成分の長さを1にする（正規化）
        norm = np.linalg.norm(w)
        if norm > 1e-10:  # 零ベクトルでなければ基底に追加
            basis.append(w / norm)
            
    return np.array(basis)

# --- 実行と検証 ---

# 1. ランダムな3つのベクトルを生成
np.random.seed(42)
v_random = np.random.randn(3, 3)
print("元のランダムなベクトル:\n", v_random)

# 2. グラム・シュミット法を適用
ortho_basis = gram_schmidt(v_random)
print("\n生成された正規直交基底:\n", ortho_basis)

# 3. 直交性と正規性のチェック
print("\n--- 検証結果 ---")
# 内積行列を計算 (B * B^T)
check_matrix = np.dot(ortho_basis, ortho_basis.T)

# 単位行列に近ければ、正規直交性が保たれている
print("内積行列 (単位行列に近ければ成功):\n", np.round(check_matrix, 10))

# 個別の確認
print(f"e1 と e2 の内積: {np.dot(ortho_basis[0], ortho_basis[1]):.10f}")
print(f"e1 の長さ: {np.linalg.norm(ortho_basis[0]):.10f}")