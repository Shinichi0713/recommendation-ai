import numpy as np
from scipy.linalg import svd


# ソフト閾値処理
def shrink(x, kappa):
    """
    Soft-thresholding operator (element-wise)
    shrink(x, kappa) = sign(x) * max(|x| - kappa, 0)
    """
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)

# 各ノルムの計算
def prox_nuclear_norm(L, tau):
    """
    Proximal operator for the nuclear norm (singular value thresholding)
    prox_nuclear_norm(L, tau) = U * shrink(S, tau) * V^T
    where L = U * S * V^T is the SVD of L
    """
    U, S, Vt = svd(L, full_matrices=False)
    # Sの特異値をソフト閾値処理
    S_shrinked = shrink(S, tau)
    # 新しい行列を再構築
    return U @ np.diag(S_shrinked) @ Vt

# 離散的にX軸方向の勾配を計算
def grad_x(img):
    h, w = img.shape
    d = np.zeros_like(img)
    d[:-1, :] += img[1:, :] - img[:-1, :]
    return d

# 離散的にY軸方向の勾配を計算
def grad_y(img):
    h, w = img.shape
    d = np.zeros_like(img)
    d[:, :-1] += img[:, 1:] - img[:, :-1]
    return d

# 勾配の計算
def grad(img):
    return grad_x(img), grad_y(img)


def div(p1, p2):
    """Divergence operator (adjoint of -grad)"""
    h, w = p1.shape
    d = np.zeros((h, w))
    # div = -grad^T
    d[:, 1:] -= p1[:, :-1]
    d[:, 0]  -= p1[:, 0]
    d[:, :-1] += p1[:, :-1]

    d[1:, :] -= p2[:-1, :]
    d[0, :]  -= p2[0, :]
    d[:-1, :] += p2[:-1, :]
    return d

def rpca_tv_admm(X, lambda_, mu, rho1, rho2, max_iter=1000, tol=1e-4, verbose=True):
    """
    RPCA + TV via ADMM

    Parameters
    ----------
    X : (h, w) array
        Observed image (matrix)
    lambda_ : float
        Weight for sparse term ||S||_1
    mu : float
        Weight for TV term ||Z||_1
    rho1, rho2 : float
        ADMM penalty parameters
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for primal/dual residuals
    verbose : bool
        Whether to print progress

    Returns
    -------
    L : (h, w) array
        Low-rank component
    S : (h, w) array
        Sparse component
    Zx, Zy : (h, w) arrays
        TV components (gradient of L)
    """
    h, w = X.shape
    # Initialize variables
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Zx, Zy = grad(L)  # Z = (Zx, Zy) ~ gradient of L
    U1 = np.zeros_like(X)  # dual for X = L + S
    U2x, U2y = grad(np.zeros_like(X))  # dual for Z = DL

    # Precompute some constants for L-update
    step_L = 1.0 / (rho1 + rho2 * 4)  # rough upper bound for Lipschitz constant

    history = {
        'primal_res1': [],
        'primal_res2': [],
        'dual_res1': [],
        'dual_res2': []
    }

    # 初期の DLx, DLy を計算
    DLx, DLy = grad(L)

    for k in range(max_iter):
        L_prev = L.copy()
        S_prev = S.copy()
        Zx_prev, Zy_prev = Zx.copy(), Zy.copy()

        # --- L-update: min_L ||L||_* + (rho1/2)||X - L - S + U1||^2 + (rho2/2)||DL - Z + U2||^2
        # Gradient of quadratic part w.r.t. L
        grad_quad_L = (
            -rho1 * (X - L - S + U1) +
            rho2 * div(DLx - Zx + U2x, DLy - Zy + U2y)
        )
        L_tilde = L - step_L * grad_quad_L
        L = prox_nuclear_norm(L_tilde, step_L)

        # Recompute DL after L update
        DLx, DLy = grad(L)

        # --- S-update: min_S lambda ||S||_1 + (rho1/2)||X - L - S + U1||^2
        S_tilde = X - L + U1
        S = shrink(S_tilde, lambda_ / rho1)

        # --- Z-update: min_Z mu ||Z||_1 + (rho2/2)||DL - Z + U2||^2
        Zx_tilde = DLx + U2x
        Zy_tilde = DLy + U2y
        Zx = shrink(Zx_tilde, mu / rho2)
        Zy = shrink(Zy_tilde, mu / rho2)

        # --- Dual updates
        # U1: dual for X = L + S
        r1 = X - L - S
        U1 = U1 + r1

        # U2: dual for Z = DL
        r2x = DLx - Zx
        r2y = DLy - Zy
        U2x = U2x + r2x
        U2y = U2y + r2y

        # --- Residuals and convergence check
        # Primal residuals
        primal_res1 = np.linalg.norm(r1, 'fro')
        primal_res2 = np.sqrt(np.linalg.norm(r2x, 'fro')**2 + np.linalg.norm(r2y, 'fro')**2)

        # Dual residuals (simplified)
        dual_res1 = rho1 * np.linalg.norm(L - L_prev + S - S_prev, 'fro')
        dual_res2 = rho2 * np.sqrt(
            np.linalg.norm(DLx - grad_x(L_prev), 'fro')**2 +
            np.linalg.norm(DLy - grad_y(L_prev), 'fro')**2
        )

        history['primal_res1'].append(primal_res1)
        history['primal_res2'].append(primal_res2)
        history['dual_res1'].append(dual_res1)
        history['dual_res2'].append(dual_res2)

        if verbose and (k % 100 == 0 or k < 10):
            print(f"Iter {k:4d}: "
                  f"primal1={primal_res1:.2e}, primal2={primal_res2:.2e}, "
                  f"dual1={dual_res1:.2e}, dual2={dual_res2:.2e}")

        if (primal_res1 < tol and primal_res2 < tol and
            dual_res1 < tol and dual_res2 < tol):
            if verbose:
                print(f"Converged at iteration {k}")
            break

    return L, S, Zx, Zy, history


if __name__ == "__main__":
    # ダミー画像（低ランク＋スパース＋ノイズ）
    h, w = 64, 64
    true_L = np.outer(np.sin(np.linspace(0, 2*np.pi, h)), np.cos(np.linspace(0, 2*np.pi, w)))
    true_S = np.zeros((h, w))
    true_S[20:30, 20:30] = 1.0  # スパースな異常
    noise = 0.01 * np.random.randn(h, w)
    X = true_L + true_S + noise

    # パラメータ（適宜調整）
    lambda_ = 0.06
    mu = 0.0005
    rho1 = 2.0
    rho2 = 1.0

    L, S, Zx, Zy, history = rpca_tv_admm(
        X, lambda_=lambda_, mu=mu, rho1=rho1, rho2=rho2,
        max_iter=1000, tol=1e-4, verbose=True
    )