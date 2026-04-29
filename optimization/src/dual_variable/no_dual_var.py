
import numpy as np

# ペナルティ法（双対変数なし）
def penalty_method(c, A, b, mu=10.0, max_iter=100, step=0.01):
    n = len(c)
    x = np.zeros(n)
    history_x = []
    history_obj_penalty = []
    
    for k in range(max_iter):
        # 制約違反
        violation = A @ x - b
        penalty_term = np.sum(np.maximum(0, violation)**2)
        
        # ペナルティ付き目的値
        obj_penalty = c @ x - mu * penalty_term
        history_x.append(x.copy())
        history_obj_penalty.append(obj_penalty)
        
        # 勾配 (最大化なので + 方向)
        grad = c.copy()
        for i in range(A.shape[0]):
            if violation[i] > 0:
                grad -= 2 * mu * violation[i] * A[i, :]
        x = np.maximum(0, x + step * grad)
    
    return np.array(history_x), np.array(history_obj_penalty)

# 実行
x_history, obj_history_penalty = penalty_method(c, A, b)

