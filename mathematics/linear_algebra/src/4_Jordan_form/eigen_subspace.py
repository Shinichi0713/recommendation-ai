import numpy as np

def direct_sum_decomposition(A, tol=1e-10):
    """
    実対称行列 A の固有空間への直和分解を扱う。
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        実対称行列
    tol : float
        固有値の一致判定の許容誤差
    
    Returns
    -------
    eigvals : ndarray
        相異なる固有値のリスト
    eigspaces : list of ndarray
        各固有空間の基底ベクトル（列ベクトル）を並べた行列のリスト
    P : ndarray
        直交行列（固有ベクトルを列に並べたもの）
    """
    # 固有値・固有ベクトルを計算（実対称行列なので eigh を使用）
    eigvals, P = np.linalg.eigh(A)
    
    # 固有値を丸めてグループ化
    rounded_vals = np.round(eigvals / tol) * tol
    unique_vals = np.unique(rounded_vals)
    
    eigspaces = []
    for lam in unique_vals:
        # 該当する固有値のインデックスを取得
        idx = np.where(np.abs(rounded_vals - lam) < tol)[0]
        # 対応する固有ベクトルを取り出し、正規直交基底として格納
        basis = P[:, idx]
        eigspaces.append(basis)
    
    return unique_vals, eigspaces, P

def decompose_vector(x, eigspaces):
    """
    ベクトル x を各固有空間成分に分解する。
    
    Parameters
    ----------
    x : ndarray, shape (n,)
        分解したいベクトル
    eigspaces : list of ndarray
        direct_sum_decomposition で得られた固有空間基底のリスト
    
    Returns
    -------
    components : list of ndarray
        各固有空間への射影成分
    """
    components = []
    for basis in eigspaces:
        # 固有空間への直交射影: proj = basis @ basis.T @ x
        proj = basis @ (basis.T @ x)
        components.append(proj)
    return components

def check_direct_sum(x, components, tol=1e-10):
    """
    直和分解の条件（和が元のベクトルに一致、成分が一意）を確認する。
    """
    # 和が元のベクトルに一致するか
    reconstructed = sum(components)
    error = np.linalg.norm(x - reconstructed)
    print(f"再構成誤差 (||x - sum components||): {error:.2e}")
    
    # 成分が互いに直交しているか（直和の特徴）
    for i in range(len(components)):
        for j in range(i+1, len(components)):
            ip = np.dot(components[i], components[j])
            if np.abs(ip) > tol:
                print(f"警告: 成分 {i} と {j} が直交していません (内積 = {ip:.2e})")
            else:
                print(f"成分 {i} と {j} は直交しています (内積 = {ip:.2e})")

# --- 例: 3x3 実対称行列 ---
A = np.array([[4, 1, 1],
              [1, 3, 0],
              [1, 0, 3]])

print("行列 A:")
print(A)
print()

# 固有空間への直和分解
eigvals, eigspaces, P = direct_sum_decomposition(A)

print("相異なる固有値:")
for lam in eigvals:
    print(f"λ = {lam:.4f}")
print()

print("各固有空間の次元:")
for i, basis in enumerate(eigspaces):
    print(f"E(λ={eigvals[i]:.4f}) の次元: {basis.shape[1]}")
print()

# 任意のベクトルを分解
x = np.array([1, 2, 3], dtype=float)
print(f"分解したいベクトル x = {x}")
components = decompose_vector(x, eigspaces)

print("\n各固有空間成分:")
for i, comp in enumerate(components):
    print(f"E(λ={eigvals[i]:.4f}) 成分: {comp}")

# 直和条件の確認
print("\n--- 直和条件の確認 ---")
check_direct_sum(x, components)