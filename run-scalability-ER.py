import numpy as np
import time
from itertools import product
from pathlib import Path

import sys
sys.path.append("./src")
from full_pipeline import run_experiment_and_plot
from viz import *

only_plot = False
debug     = False
skip_plot = True # Viz is separated

trials   = 5        # number of Monte Carlo runs
lam      = 0.1       # ridge regularization λ
delta    = 1e-3      # confidence parameter
T        = 10_000    # total horizon
T1       = int(np.sqrt(T))   # Stage-1 budget

NODES = [8, 16, 32, 64, 128, 256, 512, 1024][::-1]  # or [10, 25, 50, 100, 200, 400, 800, 1000]
PROBS  = [.1,]
NOISES = [.1,]
ARMS   = [2,]
N_ARMS = [100,]

def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

# Precompute all combinations so we can track progress
combos = list(product(NODES, PROBS, NOISES, ARMS, N_ARMS))
total = len(combos)
start = time.perf_counter()

# --- experiment setup summary ---
print("-"*25)
print("\nStarting experiments with:")
print(f"  trials   = {trials}")
print(f"  lam      = {lam}")
print(f"  delta    = {delta}")
print(f"  T        = {T}")
print(f"  T1       = {T1}")
print("  search space:")
print(f"    NODES  = {NODES}")
print()

for idx, (n, p, sigma, a, n_arms) in enumerate(combos, start=1):
    n_e = int(a * n)  # number of changes in each arm w.r.t initial graph
    save_to = f"./figures/scalability_ER_n{n}_p{int(100*p)}_sigma{int(100*sigma)}_arms{n_arms}_{n_e}.pdf"

    # --- run experiment ---
    if n <= 64:
        benchmark = {"OUR", "OFUL"}
    else:
        benchmark = {"OUR"}

    run_experiment_and_plot(
        n=n, p=p, T=T, T1=T1, trials=trials, n_e=n_e,
        sigma=sigma, lam=lam, delta=delta,
        save_path=save_to, debug=debug,
        only_plot=only_plot,
        skip_plot=skip_plot,
        run_efficient=True,
        benchmark=benchmark
    )

    # --- progress report ---
    elapsed = time.perf_counter() - start
    avg_per = elapsed / idx
    remaining = total - idx
    eta = avg_per * remaining

    print(
        f"[{idx}/{total}] "
        f"elapsed: {_fmt_time(elapsed)} | "
        f"eta: {_fmt_time(eta)} | "
        f"remaining: {remaining} combos | "
        f"current: n={n}, p={p}, sigma={sigma}, a={a}, n_arms={n_arms}",
        flush=True
    )

print("\nFinished ✅")
print("\n", "-"*25)



# =============================================================
# ========================== PLOT =============================

    
plot_scalability_curve(collect_rows_scalability(), Path("./figures") / "scalability_ER.pdf")