import numpy as np
import matplotlib.pyplot as plt

# 問題データ
c = np.array([3.0, 2.0])
A = np.array([[1, 1],
              [1, 0]])
b = np.array([4.0, 3.0])

# 双対上昇法（双対変数あり）
def dual_ascent(c, A, b, max_iter=100, step=0.1):
    m = A.shape[0]
    lam = np.zeros(m)  # 双対変数 λ
    history_lam = []
    history_obj = []
    
    for k in range(max_iter):
        # x(λ) = argmin_{x>=0} L(x,λ)
        # ∇_x L = c + A^T λ
        grad_x = c + A.T @ lam
        # 単純な射影: x = max(0, - (c + A^T λ)) などではなく、
        # ここでは簡略化のため線形計画ソルバを使わず、閉形式で近似
        # 実際には scipy.optimize.linprog などを使うべきですが、
        # ここでは概念を示すために簡略化します。
        x = np.maximum(0, -grad_x)  # これはあくまで概念的な近似
        
        # 双対目的値: min_x L(x,λ) の近似
        dual_obj = c @ x + lam @ (A @ x - b)
        
        history_lam.append(lam.copy())
        history_obj.append(dual_obj)
        
        # 双対変数の更新 (subgradient ascent)
        # g = A x - b が双対目的の劣勾配
        g = A @ x - b
        lam = np.maximum(0, lam + step * g)
    
    return np.array(history_lam), np.array(history_obj)

# 実行
lam_history, obj_history_dual = dual_ascent(c, A, b)