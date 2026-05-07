import numpy as np
from scipy.fft import fft2, ifft2

# -----------------------------
# 基本演算
# -----------------------------
def soft_thresholding(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

def svt(X, tau):
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    return (U * s) @ Vh

# -----------------------------
# 3D gradient（周期境界）
# -----------------------------
def gradient_3d(L):
    gx = np.roll(L, -1, axis=1) - L  # x方向
    gy = np.roll(L, -1, axis=0) - L  # y方向
    gz = np.roll(L, -1, axis=2) - L  # スペクトル方向
    return gx, gy, gz

def divergence_3d(px, py, pz):
    return (
        (np.roll(px, 1, axis=1) - px) +
        (np.roll(py, 1, axis=0) - py) +
        (np.roll(pz, 1, axis=2) - pz)
    )

# -----------------------------
# メイン：TV-RPCA (3D TV)
# -----------------------------
def tv_rpca_3d_admm(M, lambd=0.01, gamma=0.01, rho=1.0,
                   max_iter=200, tol=1e-5, verbose=False):
    """
    M: (H, W, B)
    """

    H, W, B = M.shape
    eps = 1e-12

    # 変数
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    J = np.zeros_like(M)

    Gx = np.zeros_like(M)
    Gy = np.zeros_like(M)
    Gz = np.zeros_like(M)

    # dual
    U1 = np.zeros_like(M)
    U2 = np.zeros_like(M)
    U3x = np.zeros_like(M)
    U3y = np.zeros_like(M)
    U3z = np.zeros_like(M)

    # -----------------------------
    # FFT用カーネル（空間のみ）
    # -----------------------------
    dx = np.zeros((H, W))
    dx[0, 0] = 1; dx[0, 1] = -1

    dy = np.zeros((H, W))
    dy[0, 0] = 1; dy[1, 0] = -1

    lap_xy = np.abs(fft2(dx))**2 + np.abs(fft2(dy))**2  # (H,W)

    # スペクトル方向のラプラシアン（周期）
    k = np.arange(B)
    lap_z = 2 - 2*np.cos(2*np.pi*k / B)   # (B,)

    # broadcast用
    denom = 2.0 + lap_xy[:, :, None] + lap_z[None, None, :]
    denom = np.maximum(denom, eps)

    normM = np.linalg.norm(M) + eps

    for it in range(max_iter):
        L_prev = L.copy()

        # -------------------------
        # S更新
        # -------------------------
        S = soft_thresholding(M - L + U1, lambd / rho)

        # -------------------------
        # J更新（低ランク）
        # reshape: (HW, B)
        # -------------------------
        L_mat = (L + U2).reshape(H*W, B)
        J_mat = svt(L_mat, 1.0 / rho)
        J = J_mat.reshape(H, W, B)

        # -------------------------
        # G更新（3D TV）
        # -------------------------
        gx, gy, gz = gradient_3d(L)

        Gx = soft_thresholding(gx + U3x, gamma / rho)
        Gy = soft_thresholding(gy + U3y, gamma / rho)
        Gz = soft_thresholding(gz + U3z, gamma / rho)

        # -------------------------
        # L更新
        # -------------------------
        rhs = (
            (M - S + U1)
            + (J - U2)
            + divergence_3d(Gx - U3x, Gy - U3y, Gz - U3z)
        )

        # FFT (空間)
        RHS_fft = fft2(rhs, axes=(0,1))
        L = np.real(ifft2(RHS_fft / denom, axes=(0,1)))

        # -------------------------
        # dual更新
        # -------------------------
        r1 = M - L - S
        r2 = L - J
        gx, gy, gz = gradient_3d(L)

        r3x = gx - Gx
        r3y = gy - Gy
        r3z = gz - Gz

        U1 += r1
        U2 += r2
        U3x += r3x
        U3y += r3y
        U3z += r3z

        # -------------------------
        # 収束判定
        # -------------------------
        primal = np.sqrt(
            np.linalg.norm(r1)**2 +
            np.linalg.norm(r2)**2 +
            np.linalg.norm(r3x)**2 +
            np.linalg.norm(r3y)**2 +
            np.linalg.norm(r3z)**2
        ) / normM

        rel_L = np.linalg.norm(L - L_prev) / (np.linalg.norm(L_prev) + eps)

        if verbose:
            print(f"{it:03d}  primal={primal:.3e}  dL={rel_L:.3e}")

        if primal < tol and rel_L < tol:
            break

    return L, S