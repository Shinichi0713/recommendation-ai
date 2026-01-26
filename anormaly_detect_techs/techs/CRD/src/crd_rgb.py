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

def crd_anomaly_detection(image_gray, win_out=9, win_in=3, lamda=0.01):
    h, w = image_gray.shape
    # スコアを格納する行列
    anomaly_map = np.zeros((h, w))

    # 窓の半径を計算
    r_out = win_out // 2
    r_in = win_in // 2

    # パディング（端の画素も処理できるようにする）
    img_pad = np.pad(image_gray, r_out, mode='edge')

    for i in range(r_out, h + r_out):
        for j in range(r_out, w + r_out):
            # ターゲット画素
            target_vec = img_pad[i, j].reshape(-1, 1)

            # 周囲の背景画素を辞書行列として収集
            dictionary_pixels = []
            for m in range(i - r_out, i + r_out + 1):
                for n in range(j - r_out, j + r_out + 1):
                    if abs(m - i) > r_in or abs(n - j) > r_in:
                        dictionary_pixels.append(img_pad[m, n])
            dictionary_matrix = np.array(dictionary_pixels).reshape(-1, len(dictionary_pixels)).T

            # CRDスコアの計算
            score = multivariate_crd_score(target_vec, dictionary_matrix, lamda)
            anomaly_map[i - r_out, j - r_out] = score

        return anomaly_map
    
if __name__ == "__main__":
    # テスト用コード
    import cv2
    import matplotlib.pyplot as plt

    # グレースケール画像の読み込み
    image_path = 'test_image.png'  # 適切な画像パスに変更してください
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # CRD異常検知の実行
    anomaly_map = crd_anomaly_detection(image_gray, win_out=9, win_in=3, lamda=0.01)

    # 結果の表示
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Anomaly Map (CRD)')
    plt.imshow(anomaly_map, cmap='hot')
    plt.axis('off')

    plt.show()