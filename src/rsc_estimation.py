import numpy as np

# ==============================
# Gram Matrix Eigenvalue
# ==============================
# Gram matrix eigenvalue = global minimum curvature (possibly very small or zero if some direction is completely invisible).

def min_eig_gram(arms):
    # arms: list of (d,d)
    A = np.array([a.flatten() for a in arms])  # shape (N, d^2)
    H = (A.T @ A) / A.shape[0]                 # Gram matrix (d^2 x d^2)
    w = np.linalg.eigvalsh(H)
    return w[0]   # smallest eigenvalue



# ==============================
# Projected GD (rank-2) with cone enforcement
# ==============================
# PGD restricted search = curvature along the statistically relevant set of directions.

# ---------- small helpers ----------
def sym(A): return 0.5*(A + A.T)
def fro_norm(A): return np.linalg.norm(A, 'fro')
def normalize_f(A): 
    n = fro_norm(A);  return (A/n if n > 1e-15 else A)

def rank2_project(A):
    A = sym(A)
    w, V = np.linalg.eigh(A)
    idx = np.argsort(np.abs(w))[-2:]         # top-2 by |eig|
    return sym((V[:, idx] * w[idx]) @ V[:, idx].T)

# ---------- cone pieces (optional) ----------
def _split_overlineM(Delta, s):
    s = s.reshape(-1,1);  P = s @ s.T;  I = np.eye(len(s));  Pp = I - P
    Dp = Pp @ Delta @ Pp                  # in overline M^\perp
    Db = Delta - Dp                       # in overline M
    return Db, Dp

def _nuc(A): return float(np.sum(np.linalg.svd(A, compute_uv=False)))

def cone_enforce(Delta, s, ratio=3.0):
    Db, Dp = _split_overlineM(Delta, s)
    nb, npn = _nuc(Db), _nuc(Dp)
    if nb < 1e-15 or npn <= ratio*nb + 1e-15: 
        return sym(Delta)
    return sym(Db + (ratio*nb/npn)*Dp)

# ---------- vectorized Q and grad ----------
def Q_value(Delta, X):           # X shape: (N,d,d)
    a = np.einsum('nij,ij->n', X, Delta, optimize=True)
    return float(np.mean(a*a))

def grad_Q(Delta, X):
    a = np.einsum('nij,ij->n', X, Delta, optimize=True)  # (N,)
    G = np.einsum('n,nij->ij', a, X, optimize=True)      # sum_i a_i X_i
    return sym(2.0 * G / X.shape[0])

# ---------- main: minimal PGD ----------
def kappa0_cone_fast(
    Xs,                     # list/array of N (d x d) symmetric arms
    s=None,                 # unit vector or None (skip cone)
    n_restarts=15,
    max_iter=200,
    step0=1.0,              # initial step size
    backtrack=0.5,          # Armijo shrink
    tol=1e-9,
    dtype=np.float32,
    minibatch=None,         # e.g. 256 for stochastic gradient, else None
    verbose=False
):
    X = np.asarray(Xs, dtype=dtype)                     # (N,d,d)
    N, d = X.shape[0], X.shape[1]
    rng = np.random.default_rng(0)

    def grad(Delta):
        if minibatch is None or minibatch >= N:
            return grad_Q(Delta, X)
        idx = rng.integers(0, N, size=minibatch)
        return grad_Q(Delta, X[idx])

    best_val, best_D = np.inf, None
    for r in range(n_restarts):
        # random rank-2 init, ||Δ||_F=1
        U, _ = np.linalg.qr(rng.standard_normal((d,2)).astype(dtype))
        lam = rng.standard_normal(2).astype(dtype); lam /= (np.linalg.norm(lam) + 1e-15)
        Delta = normalize_f(sym((U * lam) @ U.T)).astype(dtype)
        val = Q_value(Delta, X); step = step0

        for it in range(max_iter):
            G = grad(Delta).astype(dtype)

            # backtracking Armijo
            while True:
                Dtmp = Delta - step * G
                Dtmp = rank2_project(Dtmp)
                if s is not None: Dtmp = cone_enforce(Dtmp, s, ratio=3.0)
                Dtmp = normalize_f(Dtmp).astype(dtype)

                new_val = Q_value(Dtmp, X)
                # accept if decreased or step too small
                if new_val <= val - 1e-8 * step * float(fro_norm(G)**2) or step < 1e-8:
                    break
                step *= backtrack

            Delta, val = Dtmp, new_val
            if verbose and it % 25 == 0:
                print(f"[{r+1:02d}] it {it:03d}  Q={val:.6e}  step={step:.2e}")
            if step < 1e-8 or (it > 5 and abs(new_val - val) < tol*max(1.0, abs(val))):
                break   # tiny step or stabilized

        if val < best_val:
            best_val, best_D = float(val), Delta.copy()

    return best_val, best_D




def avg_commutator_norm(Xs):
    K = len(Xs); s = 0.0; c = 0
    for i in range(K):
        for j in range(i+1, K):
            C = Xs[i] @ Xs[j] - Xs[j] @ Xs[i]
            s += np.linalg.norm(C, 'fro'); c += 1
    return s / max(c,1)