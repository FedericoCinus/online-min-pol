import numpy as np
from numpy.linalg import norm

# ---------------------------------------------------------
# Confidence radius β_t for linear bandits
# ---------------------------------------------------------
def compute_beta(t, d_feat, sigma, lambda_reg, delta, Lx, S):
    """
    Compute the confidence radius β_t for OFUL (minimization version).

    β_t = σ √( d log((λ + t Lx²)/δ) ) + √λ · S

    Args
    ----
    t          : current round index (1..T)
    d_feat     : feature dimension (length of x_i)
    sigma      : noise level in observed rewards
    lambda_reg : ridge regularization parameter λ
    delta      : failure probability (confidence 1−δ)
    Lx         : bound on arm norms (max ||x||₂)
    S          : assumed bound on parameter norm (||θ*||₂)

    Returns
    -------
    beta_t : float, confidence radius
    """
    return sigma * np.sqrt(
        d_feat * np.log((lambda_reg + t * (Lx ** 2)) / max(delta, 1e-12))
    ) + np.sqrt(lambda_reg) * S


# ---------------------------------------------------------
# Arm selection via lower confidence bound (LCB)
# ---------------------------------------------------------
def argmin_lcb_from_A(A, X, theta_hat, beta):
    """
    Pick the arm minimizing the LCB:
        LCB(x) = xᵀ θ̂ − β √(xᵀ A⁻¹ x)

    Args
    ----
    A         : design matrix (λI + Σ x_s x_sᵀ), shape (d, d)
    X         : feature matrix of all arms, shape (N, d)
    theta_hat : current estimate of θ*, shape (d,)
    beta      : confidence radius β_t

    Returns
    -------
    j : int
        index of the chosen arm (minimizes LCB)
    """
    # Solve A s_i = x_i for all arms simultaneously → columns of S_solve are A⁻¹ x_i
    S_solve = np.linalg.solve(A, X.T)            # (d, N)
    # Quadratic form x_iᵀ A⁻¹ x_i for all arms
    quad_all = np.einsum('nd,dn->n', X, S_solve)
    # Compute lower confidence bounds (optimistic for minimization)
    lcb_all  = (X @ theta_hat) - beta * np.sqrt(np.maximum(quad_all, 0.0))
    return int(np.argmin(lcb_all))


# ---------------------------------------------------------
# Stage 2: OFUL on reduced features (dim = 2d−1)
# ---------------------------------------------------------
def oful_rank1(x_proj, theta_proj, T2, sigma, lambda_reg, delta, cumulative_regret=0.0, rng=None, N=None, 
               return_mins=False, means=None, verbose=False):
    """
    OFUL (minimization version) restricted to the reduced feature space.

    Each arm i has feature vector x_i ∈ ℝᵖ (p = 2d−1).
    The true cost of arm i is μ_i = x_iᵀ θ_proj (fixed across time).
    The algorithm maintains a confidence set around θ* and plays the arm
    with the lowest plausible cost (LCB).

    Regret is defined as:
        Regret_t = μ(x_t) − min_j μ(x_j)

    Args
    ----
    x_proj   : (N, p) reduced feature matrix
    theta_proj : (p,) true parameter in reduced space
    T2       : number of rounds
    sigma    : noise level in rewards
    lambda_reg : ridge parameter λ
    delta    : confidence failure probability
    cumulative_regret : starting value (e.g. from stage 1)
    rng      : numpy random Generator
    N          : number of nodes


    Returns
    -------
    regrets : (T2,) array of cumulative regrets
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(x_proj)                 # (N, p)
    _, p = X.shape

    # True expected costs (environment) and best achievable cost
    if means is None:
        means = X @ theta_proj   # fall back to projected model if nothing given
    best  = float(np.min(means))

    # Constants for confidence sets
    #print("Lx", float(np.max(np.linalg.norm(X, axis=1))) + 1e-12, N)
    #print("S", float(float(norm(theta_proj))), N)
    Lx = N # upper bound     # tight bound = float(np.max(np.linalg.norm(X, axis=1))) + 1e-12
    S  = N # upper bound     # tight bound = float(norm(theta_proj))                          # ||θ*||

    # Initialize statistics
    A = lambda_reg * np.eye(p)  # design matrix
    b = np.zeros(p)             # weighted sum of observations

    regrets, cum = [], float(cumulative_regret)
    mins, current_min = [], float("inf")

    # Main OFUL loop
    for t in range(1, T2 + 1):
        # Current parameter estimate
        theta_hat = np.linalg.solve(A, b)

        # Confidence radius
        beta = compute_beta(t, p, sigma, lambda_reg, delta, Lx, S)

        # Arm selection by LCB rule
        j = argmin_lcb_from_A(A, X, theta_hat, beta)
        xj = X[j]

        # Observe noisy cost
        y = float(means[j] + rng.normal(scale=sigma))

        # Update statistics
        A += np.outer(xj, xj)
        b += xj * y

        # Regret accounting (minimization: chosen mean − best mean)
        cum += float(means[j] - best)
        regrets.append(cum)

        current_min = min(current_min, float(means[j]))
        mins.append(np.absolute(current_min)) # avoids random error below zero)

    if verbose:
        print(f"Last obj: {mins[-1]:.3f}")
    
    if return_mins:
        return np.array(mins, dtype=float)

    return np.array(regrets, dtype=float)


# ---------------------------------------------------------
# Baseline: OFUL on full d² features
# ---------------------------------------------------------
def oful_loop(T, sigma, lambda_reg, delta, arms, s_true, rng, N, verbose=False, return_mins=False):
    """
    Full-dimensional OFUL baseline (minimization version).

    Uses the full d² vectorization of arms rather than reduced 2d−1.
    Same logic as oful_rank1, but operates in higher dimension.

    Args
    ----
    T          : horizon
    sigma      : noise level
    lambda_reg : ridge parameter
    delta      : confidence failure prob
    arms       : list of (d,d) arm matrices
    s_true     : ground-truth direction (defines θ* = s sᵀ)
    rng        : numpy Generator
    N          : number of nodes

    Returns
    -------
    regrets : (T,) array of cumulative regrets
    """
    rng = np.random.default_rng() if rng is None else rng
    # Feature matrix: each arm flattened to length d²
    X = np.vstack([A.reshape(-1) for A in arms])     
    theta_true_vec = np.outer(s_true, s_true).reshape(-1)


    ##########################
    # Projection of θ* onto span(Xᵀ)
    Q, _ = np.linalg.qr(X.T, mode='reduced')            # columns span(Xᵀ)
    theta_proj = Q @ (Q.T @ theta_true_vec)
    residual = theta_true_vec - theta_proj
    rel_miss = np.linalg.norm(residual) / (np.linalg.norm(theta_true_vec) + 1e-12)

    if verbose:
        print(f"Coverage test: ||θ* - Proj_span(Xᵀ) θ*|| / ||θ*|| = {rel_miss:.3e}")
    # If rel_miss ~ 0 ⇒ statistically identifiable.
    # If rel_miss is large (e.g., > 1e-2 or 1e-1), the arms don't cover θ* well.
    ##########################

    _, d = X.shape
    means = X @ theta_true_vec
    best  = float(np.min(means))


    #print("Lx", float(np.max(np.linalg.norm(X, axis=1))) + 1e-12, N)
    #print("S", float(float(norm(theta_true_vec))), N)
    Lx = N # upper bound     # tight bound = float(np.max(np.linalg.norm(X, axis=1))) + 1e-12
    S  = N # upper bound     # tight bound =float(norm(theta_true_vec))

    A = lambda_reg * np.eye(d)
    b = np.zeros(d)

    regrets = np.zeros(T, dtype=float)
    cum = 0.0
    mins    = np.zeros(T, dtype=float)
    current_min = float("inf")


    for t in range(1, T + 1):
        theta_hat = np.linalg.solve(A, b)
        beta = compute_beta(t, d, sigma, lambda_reg, delta, Lx, S)

        j = argmin_lcb_from_A(A, X, theta_hat, beta)
        xj = X[j]

        # Observe noisy cost
        y = float(means[j] + np.random.normal(scale=sigma))

        # Update statistics
        A += np.outer(xj, xj)
        b += xj * y

        # Regret accounting
        cum += float(means[j] - best)
        regrets[t - 1] = cum

        # Running minimum of true objective
        current_min = min(current_min, float(means[j]))
        mins[t - 1] = np.absolute(current_min) # avoids random error below zero
    
    if verbose:
        print(f"Last obj: {mins[-1]:.3f}")

    if return_mins:
        return mins
    
    return regrets