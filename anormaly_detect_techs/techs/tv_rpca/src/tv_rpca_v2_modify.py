import numpy as np
from scipy.fft import fft2, ifft2

def soft_thresholding(x, tau):
    """Soft-thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

def svt(X, tau):
    """Singular Value Thresholding (prox of nuclear norm)."""
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return (U * s_thresh) @ Vh

def gradient_periodic(L):
    """
    Forward differences with periodic boundary conditions.
    Returns:
        gx: horizontal gradient
        gy: vertical gradient
    """
    gx = np.roll(L, -1, axis=1) - L
    gy = np.roll(L, -1, axis=0) - L
    return gx, gy

def divergence_periodic(px, py):
    """
    Adjoint of forward differences with periodic boundary conditions.
    This is the discrete divergence corresponding to gradient_periodic().
    """
    return (np.roll(px, 1, axis=1) - px) + (np.roll(py, 1, axis=0) - py)

def tv_rpca_admm(M, lambd=0.01, gamma=0.01, rho=1.0, max_iter=200, tol=1e-5, verbose=False):
    """
    TV-RPCA via ADMM.

    Solves:
        min_{L,S} ||L||_* + lambd ||S||_1 + gamma TV(L)
        s.t. M = L + S

    with splitting:
        J = L
        G = grad(L)

    Parameters
    ----------
    M : ndarray, shape (H, W)
        Input matrix / image.
    lambd : float
        Weight for sparse term S.
    gamma : float
        Weight for TV term.
    rho : float
        ADMM penalty parameter.
    max_iter : int
        Maximum iterations.
    tol : float
        Stopping tolerance.
    verbose : bool
        Print progress if True.

    Returns
    -------
    L : ndarray
        Low-rank component.
    S : ndarray
        Sparse component.
    history : dict
        Residual history and iteration count.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("M must be a 2D array.")

    rows, cols = M.shape
    eps = 1e-12

    # Variables
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    J = np.zeros_like(M)
    Gx = np.zeros_like(M)
    Gy = np.zeros_like(M)

    # Scaled dual variables
    U1 = np.zeros_like(M)   # for M - L - S = 0
    U2 = np.zeros_like(M)   # for L - J = 0
    U3x = np.zeros_like(M)  # for grad_x(L) - Gx = 0
    U3y = np.zeros_like(M)  # for grad_y(L) - Gy = 0

    # FFT denominator for solving (2I + grad^T grad) L = rhs
    # With periodic forward differences:
    dx = np.zeros((rows, cols))
    dx[0, 0] = 1.0
    dx[0, 1] = -1.0

    dy = np.zeros((rows, cols))
    dy[0, 0] = 1.0
    dy[1, 0] = -1.0

    lap_symbol = np.abs(fft2(dx))**2 + np.abs(fft2(dy))**2
    denom = 2.0 + lap_symbol
    denom = np.maximum(denom, eps)

    history = {
        "primal_residual": [],
        "rel_change_L": [],
    }

    normM = np.linalg.norm(M, ord="fro") + eps

    for it in range(max_iter):
        L_prev = L.copy()

        # 1) S-update
        # min lambd||S||_1 + (rho/2)||M - L - S + U1||^2
        S = soft_thresholding(M - L + U1, lambd / rho)

        # 2) J-update (nuclear norm prox)
        # min ||J||_* + (rho/2)||L - J + U2||^2
        J = svt(L + U2, 1.0 / rho)

        # 3) G-update (TV prox)
        # min gamma||G||_1 + (rho/2)||grad(L) - G + U3||^2
        gx, gy = gradient_periodic(L)
        Gx = soft_thresholding(gx + U3x, gamma / rho)
        Gy = soft_thresholding(gy + U3y, gamma / rho)

        # 4) L-update
        # min (rho/2)||M - L - S + U1||^2
        #   + (rho/2)||L - J + U2||^2
        #   + (rho/2)||grad(L) - G + U3||^2
        #
        # => (2I + grad^T grad) L = (M - S + U1) + (J - U2) + grad^T(G - U3)
        rhs = (M - S + U1) + (J - U2) + divergence_periodic(Gx - U3x, Gy - U3y)

        L = np.real(ifft2(fft2(rhs) / denom))

        # 5) Dual updates
        r1 = M - L - S
        r2 = L - J
        gx_new, gy_new = gradient_periodic(L)
        r3x = gx_new - Gx
        r3y = gy_new - Gy

        U1 += r1
        U2 += r2
        U3x += r3x
        U3y += r3y

        # Stopping criteria
        primal = np.sqrt(
            np.linalg.norm(r1, ord="fro")**2 +
            np.linalg.norm(r2, ord="fro")**2 +
            np.linalg.norm(r3x, ord="fro")**2 +
            np.linalg.norm(r3y, ord="fro")**2
        ) / normM

        rel_change_L = np.linalg.norm(L - L_prev, ord="fro") / (np.linalg.norm(L_prev, ord="fro") + eps)

        history["primal_residual"].append(primal)
        history["rel_change_L"].append(rel_change_L)

        if verbose:
            print(f"iter={it:03d}  primal={primal:.3e}  rel_change_L={rel_change_L:.3e}")

        if primal < tol and rel_change_L < tol:
            break

    history["n_iter"] = it + 1
    history["final_primal_residual"] = history["primal_residual"][-1]
    history["final_rel_change_L"] = history["rel_change_L"][-1]

    return L, S, history
