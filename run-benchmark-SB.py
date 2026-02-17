import numpy as np
import time
from itertools import product

import sys
sys.path.append("./src")
from full_pipeline import run_experiment_and_plot

only_plot = True
debug     = False

trials   = 100       # number of Monte Carlo runs
lam      = 0.1       # ridge regularization λ
delta    = 1e-3      # confidence parameter
T        = 10_000    # total horizon
T1       = int(np.sqrt(T))   # Stage-1 budget

NODES  = [8, 16]
SB_SETTINGS  = ["homo"]   # Setting: homophilic or more heterophilic
NOISES = [.1]
ARMS   = [1/2]
N_ARMS = [100,]

def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

# Precompute all combinations so we can track progress
combos = list(product(NODES, SB_SETTINGS, NOISES, ARMS, N_ARMS))
total = len(combos)
start = time.perf_counter()

# --- experiment setup summary ---
if only_plot:
    print("ONLY PLOTTING")
else:
    print("-"*25)
    print("\nStarting experiments with:")
    print(f"  trials   = {trials}")
    print(f"  lam      = {lam}")
    print(f"  delta    = {delta}")
    print(f"  T        = {T}")
    print(f"  T1       = {T1}")
    print("  search space:")
    print(f"    NODES  = {NODES}")
    print(f"    SB_SETTINGS  = {SB_SETTINGS}")
    print(f"    NOISES = {NOISES}")
    print(f"    ARMS   = {ARMS}")
    print(f"    N_ARMS = {N_ARMS}")
    print()

for idx, (n, setting, sigma, a, n_arms) in enumerate(combos, start=1):
    n_e = int(a * n)  # number of changes in each arm w.r.t initial graph
    save_to = f"./figures/opdmin_vs_oful_SB_n{n}_setting{setting}_sigma{int(100*sigma)}_arms{n_arms}_{n_e}.pdf"

    # --- run experiment ---
    run_experiment_and_plot(
        n=n, setting=setting, T=T, T1=T1, n_e=n_e, trials=trials,
        sigma=sigma, lam=lam, delta=delta,
        save_path=save_to, debug=debug,
        only_plot=only_plot
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
        f"current: n={n}, setting={setting}, sigma={sigma}, a={a}, n_arms={n_arms}",
        flush=True
    )

print("\nFinished ✅")
print("\n", "-"*25)