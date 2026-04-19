import numpy as np

def jordan_basis_nilpotent(N, tol=1e-10):
    """
    べき零行列 N のジョルダン基底を構成する。
    
    Parameters
    ----------
    N : ndarray, shape (n, n)
        べき零行列
    tol : float
        数値誤差の許容範囲
    
    Returns
    -------
    basis : list of ndarray
        ジョルダン基底ベクトル（列ベクトルのリスト）
    block_sizes : list of int
        各ジョルダンブロックのサイズ
    """
    n = N.shape[0]
    
    # べき零指数 k を求める
    k = 1
    while k <= n:
        if np.allclose(np.linalg.matrix_power(N, k), 0, atol=tol):
            break
        k += 1
    else:
        raise ValueError("N is not nilpotent within n steps")
    
    print(f"べき零指数 k = {k}")
    
    # F_m = ker(N^m) の基底を段階的に求める
    F_bases = []  # F_bases[m] が F_m の基底（列ベクトルのリスト）
    
    # m=0: F_0 = {0}
    F_bases.append([])
    
    for m in range(1, k+1):
        N_power = np.linalg.matrix_power(N, m)
        # ker(N^m) の基底を求める（null space）
        # 特異値分解を使って零空間の基底を取得
        U, s, Vh = np.linalg.svd(N_power)
        rank = np.sum(s > tol)
        null_basis = Vh[rank:].T  # 零空間の基底（列ベクトル）
        
        # F_{m-1} の基底を F_m の基底に拡張
        if m == 1:
            F_bases.append([null_basis[:, i] for i in range(null_basis.shape[1])])
        else:
            # F_{m-1} の基底を F_m の基底に埋め込む
            prev_basis = np.column_stack(F_bases[m-1])
            # F_m の基底から、F_{m-1} の張る空間と直交する部分を選ぶ
            # グラム・シュミット的な直交補空間を取る簡易版
            extended_basis = []
            for i in range(null_basis.shape[1]):
                v = null_basis[:, i]
                # v から F_{m-1} への射影を引き、残りが十分大きければ採用
                if prev_basis.size > 0:
                    proj = prev_basis @ (prev_basis.T @ v)
                    v_res = v - proj
                else:
                    v_res = v
                if np.linalg.norm(v_res) > tol:
                    extended_basis.append(v_res / np.linalg.norm(v_res))
            F_bases.append(F_bases[m-1] + extended_basis)
    
    # ジョルダン鎖を生成
    basis = []
    block_sizes = []
    
    for m in range(1, k+1):
        # F_m の基底のうち、F_{m-1} には属さない部分が B_m
        # 簡易的に、F_bases[m] の後ろ |F_bases[m]| - |F_bases[m-1]| 個を B_m とみなす
        prev_len = len(F_bases[m-1])
        B_m = F_bases[m][prev_len:]
        
        for v in B_m:
            chain = []
            for j in range(m):
                w = np.linalg.matrix_power(N, j) @ v
                chain.append(w)
            # チェーンを基底に追加（順序は N^{m-1}v, ..., v の逆順が一般的）
            basis.extend(reversed(chain))
            block_sizes.append(m)
    
    return basis, block_sizes

def print_jordan_form(basis, block_sizes, N):
    """
    ジョルダン基底に関して N の行列表示（ジョルダン標準形）を表示する。
    """
    P = np.column_stack(basis)  # 基底を列に並べた変換行列
    J = np.linalg.inv(P) @ N @ P
    
    print("変換行列 P（列がジョルダン基底）:")
    print(P.round(6))
    print("\nジョルダン標準形 J = P^{-1} N P:")
    print(J.round(6))
    print("\nジョルダンブロックのサイズ:", block_sizes)

# --- 例: 3x3 べき零行列 ---
N = np.array([[0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]], dtype=float)

print("N =")
print(N)
print()

basis, block_sizes = jordan_basis_nilpotent(N)
print_jordan_form(basis, block_sizes, N)