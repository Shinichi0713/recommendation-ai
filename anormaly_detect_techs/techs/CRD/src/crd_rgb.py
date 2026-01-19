import numpy as np

def multivariate_crd_score(target_vec, dictionary_matrix, lamda=0.01):
    """
    target_vec: [d, 1] - ターゲット画素のベクトル
    dictionary_matrix (D): [d, n] - 周囲の背景画素ベクトルを並べたもの
    """
    d, n = dictionary_matrix.shape
    
    # 1. 重みアルファの計算 (正規方程式)
    # D.T @ D は [n, n] 行列
    DtD = np.dot(dictionary_matrix.T, dictionary_matrix)
    alpha = np.linalg.solve(DtD + lamda * np.eye(n), np.dot(dictionary_matrix.T, target_vec))
    
    # 2. ターゲットの再構成
    y_hat = np.dot(dictionary_matrix, alpha)
    
    # 3. 再構成誤差（マハラノビス距離に近い指標）
    score = np.linalg.norm(target_vec - y_hat)
    
    return score