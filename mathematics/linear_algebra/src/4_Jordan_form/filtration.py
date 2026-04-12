import numpy as np
import matplotlib.pyplot as plt

def nilpotent_filtration(N, max_m=10, tol=1e-10):
    """
    べき零行列 N のフィルトレーション ker(N^m) を計算する。
    
    Parameters
    ----------
    N : ndarray, shape (n, n)
        べき零行列
    max_m : int
        計算する最大の m
    tol : float
        零判定の許容誤差
    
    Returns
    -------
    dims : list of int
        dim(ker N^m) のリスト (m=0,...,max_m)
    stable_index : int
        安定化する最小の m（べき零指数）
    """
    n = N.shape[0]
    dims = []
    
    # m=0: ker(N^0) = ker(I) = {0}
    dims.append(0)
    
    # m>=1 について計算
    for m in range(1, max_m+1):
        N_power = np.linalg.matrix_power(N, m)
        # 零行列かどうかを判定（べき零指数の確認）
        if np.allclose(N_power, 0, atol=tol):
            stable_index = m
            # これ以降はすべて dim = n
            for _ in range(m, max_m+1):
                dims.append(n)
            break
        
        # ker(N^m) の次元 = n - rank(N^m)
        rank = np.linalg.matrix_rank(N_power, tol=tol)
        dim_ker = n - rank
        dims.append(dim_ker)
    else:
        # max_m までべき零にならなかった場合
        stable_index = None
    
    return dims, stable_index

def plot_filtration(dims, stable_index=None):
    """
    フィルトレーションの次元変化をプロットする。
    """
    m_values = list(range(len(dims)))
    
    plt.figure(figsize=(8, 5))
    plt.plot(m_values, dims, 'o-', linewidth=2, markersize=6)
    plt.xlabel('$m$')
    plt.ylabel('$\dim(\ker N^m)$')
    plt.title('Nilpotent Filtration: $\dim(\ker N^m)$ vs $m$')
    plt.grid(True, alpha=0.3)
    
    if stable_index is not None:
        plt.axvline(x=stable_index, color='red', linestyle='--', 
                   label=f'Stable at m={stable_index}')
        plt.legend()
    
    plt.show()

# --- 例1: 2x2 べき零行列 ---
N1 = np.array([[0, 1],
               [0, 0]])
print("例1: N1 =")
print(N1)
dims1, idx1 = nilpotent_filtration(N1, max_m=5)
print(f"dim(ker N1^m): {dims1}")
print(f"べき零指数: {idx1}")
plot_filtration(dims1, idx1)
print()

# --- 例2: 3x3 べき零行列 ---
N2 = np.array([[0, 1, 0],
               [0, 0, 1],
               [0, 0, 0]])
print("例2: N2 =")
print(N2)
dims2, idx2 = nilpotent_filtration(N2, max_m=6)
print(f"dim(ker N2^m): {dims2}")
print(f"べき零指数: {idx2}")
plot_filtration(dims2, idx2)
print()

# --- 例3: 4x4 べき零行列（2つのジョルダンブロック） ---
N3 = np.array([
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])
print("例3: N3 =")
print(N3)
dims3, idx3 = nilpotent_filtration(N3, max_m=6)
print(f"dim(ker N3^m): {dims3}")
print(f"べき零指数: {idx3}")
plot_filtration(dims3, idx3)