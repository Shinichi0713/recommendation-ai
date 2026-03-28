import numpy as np
import matplotlib.pyplot as plt

def visualize_hermitian_action():
    # 1. エルミット行列 A を作成 (A = A^H)
    # 対角要素は実数、非対角要素は互いに複素共役
    A = np.array([[3 + 0j, 1 + 2j],
                  [1 - 2j, 5 + 0j]])

    # 2. 固有値分解
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    print(f"エルミット行列 A:\n{A}")
    print(f"\n固有値 (すべて実数のはず): {eigenvalues}")

    # 3. 可視化用のベクトル（円状に配置された多数の複素ベクトル）
    theta = np.linspace(0, 2*np.pi, 30)
    # 各要素が複素数のベクトル [z1, z2]^T を作る
    z1 = np.cos(theta) + 1j * 0
    z2 = np.sin(theta) + 1j * 0
    vectors = np.vstack([z1, z2]) # (2, 30)

    # 行列 A を作用させる
    transformed_vectors = A @ vectors

    # 4. プロット (第1コンポーネントの複素平面を表示)
    plt.figure(figsize=(12, 5))

    # --- 元のベクトル (第1次元) ---
    plt.subplot(1, 2, 1)
    plt.scatter(vectors[0].real, vectors[0].imag, c='blue', label='Original (z1)')
    plt.axhline(0, color='black', lw=1); plt.axvline(0, color='black', lw=1)
    plt.title("Original Vectors (Component 1)")
    plt.axis('equal')

    # --- 変形後のベクトル (第1次元) ---
    plt.subplot(1, 2, 2)
    plt.scatter(transformed_vectors[0].real, transformed_vectors[0].imag, c='red', label='Transformed (Az1)')
    
    # 固有ベクトルの方向を強調
    # 固有値が実数なので、固有ベクトル方向への写像は「回転」を含まない
    v1_direction = eigenvectors[0, 0]
    plt.quiver(0, 0, v1_direction.real, v1_direction.imag, scale=5, color='green', label='Eigenvector Direction')
    
    plt.axhline(0, color='black', lw=1); plt.axvline(0, color='black', lw=1)
    plt.title("Transformed Vectors (Component 1)")
    plt.axis('equal')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # 5. エルミット内積の直交性確認
    inner_product = np.vdot(eigenvectors[:, 0], eigenvectors[:, 1])
    print(f"\n固有ベクトル同士のエルミット内積: {inner_product:.6e}")

visualize_hermitian_action()