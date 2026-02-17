import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
import time
import seaborn as sns
sns.set_context("paper")
sns.set_style("whitegrid")

from stage1_subspace_estimation import *
from stage2_bandit_optimization import *
from graphs import generate_graph_laplacian, generate_arms
from viz import plot_data



# ---------- Orchestrator ----------
def run_opdmin(T, T1, sigma, lambda_reg, delta, method, arms, s_true, *, rng=None, oracle_stage1=False, run_efficient=False):
    """
    OPDMin = (Stage-1) low-rank subspace estimation  ➜  (Stage-2) OFUL in 2d−1.

    Inputs
    ------
    T           : total horizon (Stage-1 + Stage-2)
    T1          : Stage-1 pulls used to estimate the subspace (rank-1 direction)
    sigma       : observation noise std
    lambda_reg  : OFUL ridge parameter λ (used in Stage-2)
    delta       : confidence failure probability for OFUL
    method      : currently supports 'oful' (minimization via LCB)
    arms        : list of (d,d) arm matrices X_i
    s_true      : (d,) ground-truth direction; Θ* = s_true s_true^T
    rng         : optional numpy.random.Generator for reproducibility

    Returns
    -------
    dict with:
      - s_hat            : (d,) estimated rank-1 direction from Stage-1
      - regrets_stage1   : (T1,) cumulative regret (minimization) from random exploration
      - regrets_stage2   : (T2,) cumulative regret from OFUL in reduced space
      - x_proj           : (N, 2d−1) projected features of the arms
      - theta_proj       : (2d−1,) projected true parameter
      - diagnostics      : dict with useful scalars (see below)
    """
    rng = np.random.default_rng() if rng is None else rng
    _d = arms[0].shape[0]
    N = len(arms)
    T2 = int(T - T1)

    # True parameter in the original (d^2) space
    theta_true_vec = np.outer(s_true, s_true).reshape(-1)

    # ---------------------------
    # Stage 1: subspace estimate
    # ---------------------------
    # LowESTR (nuclear-norm LS with SVT) → Θ̂  → leading eigenvector s_hat
    if not oracle_stage1:
        s_hat, regrets_stage1 = estimate_subspace(arms, s_true, T1, sigma, rng=rng)
    else:
        s_hat, regrets_stage1 = s_true, [0]*T1

    # Optional sanity: cosine alignment between s_hat and s_true (∈[0,1])
    cos_align = float(
        abs(np.dot(s_hat, s_true)) / (np.linalg.norm(s_hat) * np.linalg.norm(s_true) + 1e-12)
    )





    # ---------------------------------------------------
    # Projection of arms and parameter to the (2d−1) space
    # ---------------------------------------------------

    # --- Low-memory method (authoritative) ---
    x_proj, U, theta_proj_fn = project_arms_rank1_2d_minus_1_lowmem(arms, s_hat)
    if run_efficient:
        del arms
        import gc; gc.collect()
    Theta_true     = np.outer(s_true, s_true)
    theta_proj = theta_proj_fn(Theta_true)                 # (2d−1,)

    x_proj = x_proj.astype(np.float32, copy=False) # Reduce memory impact
    theta_proj = theta_proj.astype(np.float32, copy=False)


    means = x_proj @ theta_proj

    # Environment means in reduced space (time-independent)
    true_means = means                                  # (N,)
    best_val   = float(np.min(true_means))
    order      = np.argsort(true_means)
    gap12      = float(true_means[order[1]] - true_means[order[0]]) if N >= 2 else 0.0





    # -------------------------
    # Stage 2: reduced-OFUL LCB
    # -------------------------
    if method == 'oful':
        regrets_stage2 = oful_rank1(
            x_proj, theta_proj, T2, sigma, lambda_reg, delta,
            cumulative_regret=regrets_stage1[-1] if len(regrets_stage1) else 0.0,
            rng=rng,
            N=len(s_true)
        )
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Diagnostics to help debug behavior (optional but handy)
    diagnostics = {
        "cosine_s_hat_s_true": cos_align,                  # alignment quality after Stage-1
        "proj_best_mean": best_val,                        # min_i μ_i in reduced space
        "proj_gap_2nd_minus_best": gap12,                  # separation of top arms
        "stage1_regret": float(regrets_stage1[-1]) if len(regrets_stage1) else 0.0,
        "stage2_added_regret": float(regrets_stage2[-1] - (regrets_stage2[0] if len(regrets_stage2) else 0.0))
    }

    return {
        "s_hat": s_hat,
        "regrets_stage1": regrets_stage1,
        "regrets_stage2": regrets_stage2,
        "x_proj": x_proj,
        "theta_proj": theta_proj,
        "diagnostics": diagnostics,
    }





# ---------- Run Experiments ----------

def run_experiment_and_plot(n=None, p=None, setting=None, name=None, T=1, T1=1, 
                            n_e=2, n_arms=1000, trials=1, 
                            sigma=0.005, lam=1.0, delta=1e-2,
                            save_path="opdmin_vs_oful.png", 
                            debug=False, verbose=False,
                            only_plot=False,
                            benchmark={"OFUL", "ORACLE", "OUR"},
                            skip_plot=False,
                            save=True,
                            show=True,
                            run_efficient=False,
                            pol=1, # polarization exponent
):
    """
    Run OPDMin (Stage-1 LowESTR + projection + Stage-2 OFUL in 2d−1) vs
    a full d^2 OFUL baseline, and plot mean ± std cumulative regret.

    Args
    ----
    n, p         : ER(n, p) graph to build base Laplacian L
    T            : total horizon
    T1           : Stage-1 budget (random exploration + LowESTR)
    n_e          : number of changes in graph (~arm) w.r.t. initial graph
    n_arms       : number of arms
    trials       : number of i.i.d. runs to average
    sigma        : noise std in rewards
    lam          : OFUL ridge parameter λ
    delta        : confidence failure probability
    save_path    : where to save the plot
    debug        : print per-trial diagnostics if True
    """
    if only_plot:
        # ----- plot -----
        plot_data(n, save_path)
        return 

    # Allocate arrays for regrets and timings
    R_opd = np.zeros((trials, T), dtype=float)   # OPDMin: concat(Stage1, Stage2)
    R_ofu = np.zeros((trials, T), dtype=float)   # Full OFUL baseline
    R_oracle = np.zeros((trials, T), dtype=float)   # Perfect oracle in stage 1
    t_opd, t_ofu = [], []

    for trial_idx in range(trials):
        rng = np.random.default_rng(trial_idx)

        # ----- graph + arms -----
        L = generate_graph_laplacian(n=n, p=p, setting=setting, name=name)
        arms = generate_arms(L, n_e, n_arms)
        n = L.shape[0]

        # ----- choose ground-truth direction s_true (rank-1 Θ* = s s^T) -----
        # Here: random zero-mean, standardized vector
        s_true = np.random.randn(n)
        polarization_f = lambda x, pol: np.power(np.absolute(x), 1 / pol) if x >= 0 else -np.power(np.absolute(x), 1 / pol)
        polarization_f = np.vectorize(polarization_f)
        s_true = polarization_f(s_true, pol)
        s_true -= s_true.mean()
        s_true /= np.std(s_true) + 1e-12
        

        if debug:
            print(" ---> L:\n", L)
            print(" ---> s:\n", s_true)
            # ---------------------------------------------------------
            # Quick environment check in the ORIGINAL space (not reduced)
            vals = [s_true @ (X @ s_true) for X in arms]
            print("Means (original space):", vals)
            print("Gap (max - min):", max(vals) - min(vals))

            # 0) Scale of arms and responses
            print("max ||X_t||_F:", max(np.linalg.norm(X, 'fro') for X in arms))
            print("std(y):", float(np.std(s_true)))

            # 1) Is θ* in the span of the *sampled* arms?
            Xmat = np.vstack([A.reshape(-1) for A in arms]) / np.sqrt(max(T1,1))
            theta_true_vec = np.outer(s_true, s_true).reshape(-1)
            Q, _ = np.linalg.qr(Xmat.T, mode='reduced')
            proj = Q @ (Q.T @ theta_true_vec)
            rel_miss = np.linalg.norm(theta_true_vec - proj) / (np.linalg.norm(theta_true_vec)+1e-12)
            print("Stage-1 coverage: ||θ* - Proj span(X^T)|| / ||θ*|| =", rel_miss)

            # 2) Proper Lipschitz L (use spectral norm)
            G = Xmat.T @ Xmat                    # = (1/T1) sum vec(X_t) vec(X_t)^T
            L_spec = float(np.linalg.eigvalsh(G).max())
            print("L (spec) =", L_spec)


            # ----- RSC checks (symmetric arms) -----
            def svec(M):
                """Half-vectorization with sqrt(2) on off-diagonals so that
                <A,B>_F = svec(A)^T svec(B)."""
                d = M.shape[0]
                out = []
                for i in range(d):
                    out.append(M[i, i])
                    for j in range(i+1, d):
                        out.append(np.sqrt(2.0) * M[i, j])
                return np.array(out, dtype=float)

            def gram_from_arms_symmetric(arms):
                """G = (1/T1) sum_t svec(X_t) svec(X_t)^T, PSD in R^{m} with m=d(d+1)/2."""
                X = np.vstack([svec(0.5*(A + A.T)) for A in arms])  # (T1, m)
                T1 = X.shape[0]
                return (X.T @ X) / T1, X

            # Build Gram and test global RSC (τ=0)
            G, Xs = gram_from_arms_symmetric(arms)
            evals = np.linalg.eigvalsh(G)
            tol = 1e-10
            lam_min = float(evals[0])
            lam_pos_min = float(next((e for e in evals if e > tol), 0.0))  # smallest *positive* eig

            ok_global = lam_min > tol
            print(f"Check global RSC on symmetric space with τ=0: λ_min(G)={lam_min:.3e}  {'✅' if ok_global else '❌'}")

            if not ok_global:
                # Suggest an RSC-with-tolerance pair on a rank-r cone:
                rank = 1  # change if you target rank r>1
                kappa = lam_pos_min if lam_pos_min > 0 else 0.0
                tau   = (kappa / (2*rank)) if kappa > 0 else np.inf
                if np.isfinite(tau):
                    print(f"Restricted RSC (rank≤{rank}) candidate: κ={kappa:.3e}, τ={tau:.3e}  "
                        f"(ensures κ/2 - τ·r > 0) ✅")
                else:
                    print("Arms give ~no curvature on the measured span; need more diverse arms. ❌")
        # ---------------------------------------------------------

        # ---------- Baseline with perfect stage 1 ----------
        if "ORACLE" in benchmark:
            out = run_opdmin(T, T1, sigma, lam, delta, method='oful', arms=arms, s_true=s_true, rng=rng, oracle_stage1=True, run_efficient=run_efficient)
            R_oracle[trial_idx] = np.concatenate([out["regrets_stage1"], out["regrets_stage2"]]) #Concatenate Stage-1 and Stage-2 regrets
        else:
            R_oracle = None

        # ---------- OPDMin (Stage1 + projection + OFUL in 2d-1) ----------
        if "OUR" in benchmark:
            t0 = time.time()
            out = run_opdmin(T, T1, sigma, lam, delta, method='oful', arms=arms, s_true=s_true, rng=rng, run_efficient=run_efficient)
            t_opd.append(time.time() - t0)            
            R_opd[trial_idx] = np.concatenate([out["regrets_stage1"], out["regrets_stage2"]]) #Concatenate Stage-1 and Stage-2 regrets

            if debug:
                diag = out.get("diagnostics", {})
                cos_sim = diag.get("cosine_s_hat_s_true", np.nan)
                gap12  = diag.get("proj_gap_2nd_minus_best", np.nan)
                s1R    = diag.get("stage1_regret", np.nan)
                s2add  = diag.get("stage2_added_regret", np.nan)
                print(f"[Trial {trial_idx}] cos(s_hat, s_true)           = {cos_sim:.6f}")
                print(f"[Trial {trial_idx}] projected gap(second-best−best) = {gap12:.6f}")
                print(f"[Trial {trial_idx}] Stage-1 regret jump             = {s1R:.6f}")
                print(f"[Trial {trial_idx}] Stage-2 added regret            = {s2add:.6f}")
            
            if verbose:
                print(f"[Trial {trial_idx}/{trials}] OUR time: {t_opd[-1]:.3f}s\n")
        else:
            R_opd, t_opd = None, None

        # ---------- Full d^2 OFUL baseline (full features) ----------
        if "OFUL" in benchmark:
            t0 = time.time()
            # Baseline uses same oful_loop signature: (T, sigma, λ, δ, arms, s_true, rng)
            R_ofu[trial_idx] = oful_loop(T, sigma, lam, delta, arms, s_true, rng=rng, N=len(s_true))
            t_ofu.append(time.time() - t0)

            if verbose:
                print(f"[Trial {trial_idx}/{trials}] OUR time: {t_opd[-1]:.3f}s | OFUL time: {t_ofu[-1]:.3f}s\n")
        else:
            R_ofu, t_ofu = None, None
    
    # ----- aggregate -----
    if save:
        save_data(R_opd, R_ofu, R_oracle, t_opd, t_ofu, T, n, save_path)

    # ----- plot -----
    if not skip_plot:
        plot_data(n, save_path, show=show)



def save_data(R_opd, R_ofu, R_oracle, t_opd, t_ofu, T, n, orig_save_path: str):
    # 1) Resolve output locations
    save_path = orig_save_path.format(n=n) if "{n}" in orig_save_path else orig_save_path
    save_path = Path(save_path)
    data_dir = save_path.with_suffix("")  # folder with same name as file, no extension
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 2) Persist raw data + meta + config
    if R_opd is not None and len(R_opd) > 0: np.save(data_dir / "R_opd.npy", np.asarray(R_opd))
    if R_ofu is not None and len(R_ofu) > 0: np.save(data_dir / "R_ofu.npy", np.asarray(R_ofu))
    if R_oracle is not None and len(R_oracle) > 0: np.save(data_dir / "R_oracle.npy", np.asarray(R_oracle))


    if t_opd is not None and len(t_opd) > 0:
        t_opd = np.asarray(t_opd)
        lab_opd = f"Ours ($2|V|-1$) [{np.mean(t_opd):.2f}±{np.std(t_opd):.2f}s]"
    else:
        t_opd = np.array([])
        lab_opd = f"Ours ($2|V|-1$)"

    if t_ofu is not None and len(t_ofu) > 0:
        t_ofu = np.asarray(t_ofu)
        lab_ofu = f"OFUL ($|V|^2$) [{np.mean(t_ofu):.2f}±{np.std(t_ofu):.2f}s]"
    else:
        t_ofu = np.array([])
        lab_ofu = f"OFUL ($|V|^2$)"

    np.savez(data_dir / "meta.npz", T=int(T), n=n, t_opd=np.asarray(t_opd), t_ofu=np.asarray(t_ofu))


    lab_oracle = "Oracle Subspace OFUL"
    with open(data_dir / "labels.json", "w") as f:
        json.dump(
            {"lab_opd": lab_opd, "lab_ofu": lab_ofu, "lab_oracle": lab_oracle},
            f, indent=2,
        )