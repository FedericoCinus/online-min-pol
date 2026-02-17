import numpy as np
from numpy.linalg import norm, eigh, matrix_rank


# ---------- Utilities ----------
def vec(M): 
    return M.reshape(-1)

def frob_inner(A, B):
    return np.trace(A.T @ B)


# ---------- Stage 1: subspace estimation ----------
def estimate_subspace(arms, s_true, T1, sigma, lam_nuc=None, rng=None,
                      max_iter=800, tol=1e-6, verbose=False):
    rng = np.random.default_rng() if rng is None else rng
    N = len(arms)
    d = arms[0].shape[0]
    assert all(A.shape == (d, d) for A in arms)

    # ---- nuclear–norm weight (your choice) ----
    if lam_nuc is None:
        lam_nuc = 2 * np.sqrt(np.log(2 * d / 1e-2)) / np.sqrt(T1)

    Theta_true = np.outer(s_true, s_true)
    all_vals = np.array([frob_inner(X, Theta_true) for X in arms], dtype=float)
    best_val = float(np.min(all_vals))

    # ---- sample T1 arms (original, unprojected) ----
    idxs = rng.choice(N, size=T1, replace=(T1 > N))

    # ------------------- NEW (only Stage-1) -------------------
    # Project out the constant/vector-of-ones mode: P = I - hh^T
    h = np.ones(d) / np.sqrt(d)
    P = np.eye(d) - np.outer(h, h)

    # Use projected matrices ONLY in the convex optimization
    X_list_proj = [P @ arms[i] @ P for i in idxs]
    # Keep the labels from the original environment (equivalently the same
    # if s_true ⟂ 1; otherwise this keeps the problem definition unchanged)
    y = np.array([frob_inner(arms[i], Theta_true) + rng.normal(scale=sigma) for i in idxs])
    # ----------------------------------------------------------

    # ---- stage-1 cumulative regret during random exploration (unchanged) ----
    regrets, cum = [], 0.0
    for i in idxs:
        cum += float(all_vals[i] - best_val)
        regrets.append(cum)

    # ------------ Prox-grad with SVT on projected features ------------
    # L ≤ (1/T1) Σ ||X_t||_F^2  (use projected X_t here)
    L = (sum(np.linalg.norm(X, 'fro')**2 for X in X_list_proj) / max(T1, 1)) + 1e-12
    eta = 1.0 / L

    Theta = np.zeros((d, d), dtype=float)
    prev = Theta.copy()

    def grad(Th):
        G = np.zeros_like(Th)
        for Xt, yt in zip(X_list_proj, y):     # <-- use projected Xt
            G += (frob_inner(Xt, Th) - yt) * Xt
        return G / max(T1, 1)

    def svt(M, tau): # TODO: to improve using built-in proximal operators, or replace with only top eigenvector
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        s = np.maximum(s - tau, 0.0)
        return (U * s) @ Vt

    for it in range(max_iter):
        G = grad(Theta)
        Theta = svt(Theta - eta * G, eta * lam_nuc)
        diff = np.linalg.norm(Theta - prev, 'fro') / (np.linalg.norm(prev, 'fro') + 1e-12)
        if verbose and (it % 50 == 0 or diff < tol):
            resid = np.array([yt - frob_inner(Xt, Theta) for Xt, yt in zip(X_list_proj, y)])
            obj = 0.5 * np.mean(resid**2) + lam_nuc * np.sum(np.linalg.svd(Theta, full_matrices=False)[1])
            print(f"[LowESTR] it={it} diff={diff:.3e} obj={obj:.6e}")
        if diff < tol:
            break
        prev = Theta.copy()

    # Enforce symmetry and stay in the projected subspace
    Theta = 0.5 * (Theta + Theta.T)
    Theta = P @ Theta @ P                     # <-- keep only 1^⊥ energy

    # Top eigenvector = s_hat
    w, V = eigh(Theta)
    s_hat = V[:, np.argmax(w)]
    if np.dot(s_hat, s_true) < 0:
        s_hat = -s_hat

    return s_hat, np.array(regrets, dtype=float)


# ---------- Projection to (2d-1) ----------
def project_arms_rank1_2d_minus_1(arms, s_hat):
    """
    Given arms {X_i} and the current direction s_hat (unit), build:
      x'_i = [X'_{11}; X'_{2:d,1}; X'_{1,2:d}]  in R^{2d-1},
    where X' = U^T X U and U is an orthonormal basis with first column s_hat.
    Also return proj_basis B (shape d^2 x (2d-1)) such that:
      theta_proj = B^T vec(Theta)  and   <X, Theta> = <x', theta_proj>.
    """
    d = arms[0].shape[0]
    # Build U = [s_hat, U_perp]
    s_hat = s_hat / (norm(s_hat) + 1e-12)
    # Orthonormal complement by Gram-Schmidt
    U = np.zeros((d, d))
    U[:, 0] = s_hat
    k = 1
    rng = np.random.default_rng(0)
    while k < d:
        v = rng.normal(size=d)
        for j in range(k):
            v -= np.dot(U[:, j], v) * U[:, j]
        nv = norm(v)
        if nv < 1e-10:
            continue
        U[:, k] = v / nv
        k += 1

    # Helper to reduce a rotated matrix Xp to length (2d-1)
    def reduce_from_rotated(Xp):
        return np.concatenate(([Xp[0, 0]], Xp[1:, 0], Xp[0, 1:]))

    # Project all arms
    X_proj = []
    for X in arms:
        Xp = U.T @ X @ U
        X_proj.append(reduce_from_rotated(Xp))
    X_proj = np.vstack(X_proj)  # (N, 2d-1)

    # Build linear map B so that vec(U e1 e1^T U^T) -> selects the same pieces
    # For any Theta, define Theta' = U^T Theta U. Then:
    # <X, Theta> = <X', Theta'> = [X'_{11}, X'_{2:d,1}, X'_{1,2:d}] · [Theta'_{11}, Theta'_{2:d,1}, Theta'_{1,2:d}]
    # That implies theta_proj = [Theta'_{11}; Theta'_{2:d,1}; Theta'_{1,2:d}]
    # and theta_proj = B^T vec(Theta) with B implementing Theta -> Theta'.
    # Concretely: vec(Theta') = (U^T ⊗ U^T) vec(Theta). We then pick entries for (11), (2:d,1), (1,2:d).
    d2 = d * d
    K = 2 * d - 1
    kron = np.kron(U.T, U.T)  # (d^2, d^2), maps vec(Theta) -> vec(Theta')
    # Selector rows
    sel_rows = []
    # index mapping for vec by columns: vec(M)[ (j)*d + i ] = M[i, j]
    def idx(i, j):
        return j * d + i
    # (1,1)
    e = np.zeros(d2); e[idx(0, 0)] = 1.0; sel_rows.append(e)
    # (2:d,1)
    for i in range(1, d):
        e = np.zeros(d2); e[idx(i, 0)] = 1.0; sel_rows.append(e)
    # (1,2:d)
    for j in range(1, d):
        e = np.zeros(d2); e[idx(0, j)] = 1.0; sel_rows.append(e)
    S = np.vstack(sel_rows)            # (K, d^2) selects entries from vec(Theta')
    B = kron.T @ S.T                   # B has shape (d^2, K); theta_proj = B^T vec(Theta)

    return X_proj, B



def _householder_basis(s: np.ndarray) -> np.ndarray:
    """
    Orthonormal U with first column = s/||s||.
    Low memory; deterministic; O(d^2).
    """
    s = np.asarray(s, dtype=float)
    d = s.size
    s = s / (norm(s) + 1e-12)
    e1 = np.zeros_like(s); e1[0] = 1.0
    v = s - e1
    nv = norm(v)
    if nv < 1e-12:
        U = np.eye(d)
        U[:, 0] = s
        return U
    u = v / nv
    H = np.eye(d) - 2.0 * np.outer(u, u)  # Householder: H e1 = s
    H[:, 0] = s  # pin first column exactly
    return H  # orthonormal

def project_arms_rank1_2d_minus_1_lowmem(
    arms: list[np.ndarray],
    s_hat: np.ndarray,
):
    """
    Project quadratic arms X_i (d×d) to x'_i ∈ R^{2d-1} with minimal memory.

    Returns
    -------
    X_proj : (N, 2d-1) array
    U      : (d, d) orthonormal, first col = s_hat/||s_hat||
    theta_proj_fn : callable mapping Θ (d×d) -> θ_proj (2d-1) without kron
    """
    d = arms[0].shape[0]
    U = _householder_basis(s_hat)
    u1, Urest = U[:, 0], U[:, 1:]      # first column and the rest

    # Preallocate and fill row-by-row
    N = len(arms)
    X_proj = np.empty((N, 2*d - 1), dtype=float)

    for i, X in enumerate(arms):
        # Compute only what we need: [u1^T X u1; Urest^T X u1; u1^T X Urest]
        Xu1   = X @ u1                # (d,)
        x11   = float(u1 @ Xu1)       # scalar
        XUres = X @ Urest             # (d, d-1)
        col1  = Urest.T @ Xu1         # (d-1,)
        row1  = u1 @ XUres            # (d-1,)

        X_proj[i, 0]    = x11
        X_proj[i, 1:d]  = col1
        X_proj[i, d:]   = row1

    # Tiny helper: Θ -> θ_proj (no kron, no full rotation)
    def theta_proj_fn(Theta: np.ndarray) -> np.ndarray:
        Tu1   = Theta @ u1
        t11   = float(u1 @ Tu1)
        TUres = Theta @ Urest
        col1  = Urest.T @ Tu1
        row1  = u1 @ TUres

        out = np.empty(2*d - 1, dtype=Theta.dtype)
        out[0]    = t11
        out[1:d]  = col1
        out[d:]   = row1
        return out

    return X_proj, U, theta_proj_fn
