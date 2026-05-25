import numpy as np

def cp_als(X, rank, max_iter=100, tol=1e-6):
    """
    CP分解（ランクR）を交互最小二乗法（ALS）で求める（修正版）
    """
    I, J, K = X.shape
    
    # 因子行列の初期化（乱数）
    A = np.random.randn(I, rank)
    B = np.random.randn(J, rank)
    C = np.random.randn(K, rank)
    
    # モード行列化
    X1 = X.reshape(I, -1)  # モード-1行列化 (I, J*K)
    X2 = X.transpose(1, 0, 2).reshape(J, -1)  # モード-2行列化 (J, I*K)
    X3 = X.transpose(2, 0, 1).reshape(K, -1)  # モード-3行列化 (K, I*J)
    
    for it in range(max_iter):
        # --- Aの更新（B, C固定） ---
        # Khatri-Rao積: Z = C ⊙ B, 形状 (J*K, rank)
        Z = np.zeros((J*K, rank))
        for j in range(J):
            for k in range(K):
                idx = j*K + k
                Z[idx, :] = B[j, :] * C[k, :]
        
        # 最小二乗問題: A Z^T ≈ X1  →  A ≈ X1 Z (Z^T Z)^{-1}
        # ただし数値安定のため lstsq で解く: A Z^T = X1 を A について解く
        # 形状: X1: (I, J*K), Z: (J*K, rank) → A: (I, rank)
        A_new, _, _, _ = np.linalg.lstsq(Z, X1.T, rcond=None)
        A_new = A_new.T  # (I, rank)
        
        # --- Bの更新（A, C固定） ---
        # Z = C ⊙ A_new, 形状 (I*K, rank)
        Z = np.zeros((I*K, rank))
        for i in range(I):
            for k in range(K):
                idx = i*K + k
                Z[idx, :] = A_new[i, :] * C[k, :]
        
        # B Z^T = X2 を B について解く
        # 形状: X2: (J, I*K), Z: (I*K, rank) → B: (J, rank)
        B_new, _, _, _ = np.linalg.lstsq(Z, X2.T, rcond=None)
        B_new = B_new.T  # (J, rank)
        
        # --- Cの更新（A_new, B_new固定） ---
        # Z = B_new ⊙ A_new, 形状 (I*J, rank)
        Z = np.zeros((I*J, rank))
        for i in range(I):
            for j in range(J):
                idx = i*J + j
                Z[idx, :] = A_new[i, :] * B_new[j, :]
        
        # C Z^T = X3 を C について解く
        # 形状: X3: (K, I*J), Z: (I*J, rank) → C: (K, rank)
        C_new, _, _, _ = np.linalg.lstsq(Z, X3.T, rcond=None)
        C_new = C_new.T  # (K, rank)
        
        # 形状チェック
        assert A_new.shape == (I, rank), f"A_new shape: {A_new.shape}, expected: {(I, rank)}"
        assert B_new.shape == (J, rank), f"B_new shape: {B_new.shape}, expected: {(J, rank)}"
        assert C_new.shape == (K, rank), f"C_new shape: {C_new.shape}, expected: {(K, rank)}"
        
        # 収束判定
        diff_A = np.linalg.norm(A_new - A)
        diff_B = np.linalg.norm(B_new - B)
        diff_C = np.linalg.norm(C_new - C)
        if max(diff_A, diff_B, diff_C) < tol:
            A, B, C = A_new, B_new, C_new
            print(f"収束しました（反復 {it+1} 回）")
            break
        
        A, B, C = A_new, B_new, C_new
    
    return A, B, C

def reconstruct_cp(A, B, C):
    """
    CP分解からテンソルを再構成する
    """
    I, rank = A.shape
    J = B.shape[0]
    K = C.shape[0]
    X_hat = np.zeros((I, J, K))
    for r in range(rank):
        # ランク1テンソル a_r ∘ b_r ∘ c_r を計算
        rank1 = np.einsum('i,j,k->ijk', A[:, r], B[:, r], C[:, r])
        X_hat += rank1
    return X_hat

# 小さなテンソルを作成（例：2x3x2）
X = np.array([
    [[1, 2],
     [3, 4],
     [5, 6]],
    [[7, 8],
     [9, 10],
     [11, 12]]
])

print("元のテンソル X:")
print(X)
print("形状:", X.shape)

# CP分解（ランク2）
rank = 2
A, B, C = cp_als(X, rank=rank)

print("\n因子行列 A (I x rank):")
print(A)
print("因子行列 B (J x rank):")
print(B)
print("因子行列 C (K x rank):")
print(C)

# 再構成
X_hat = reconstruct_cp(A, B, C)

print("\n再構成テンソル X_hat:")
print(X_hat)

print("\n誤差（Frobeniusノルム）:", np.linalg.norm(X - X_hat))

# ランク1成分を個別に表示
print("\n--- ランク1成分の可視化 ---")
for r in range(rank):
    comp = np.einsum('i,j,k->ijk', A[:, r], B[:, r], C[:, r])
    print(f"ランク1成分 {r+1}:")
    print(comp)
    print()