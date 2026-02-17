import numpy as np
import sys
sys.path.append("./src")

from graphs import generate_graph_laplacian, generate_arms, generate_diverse_arms
from rsc_estimation import kappa0_cone_fast, avg_commutator_norm, min_eig_gram




from pathlib import Path
import numpy as np
import pandas as pd

# -------- minimal, RSC-informative combos --------

n_restarts = 5

COMBOS = [
    ("ER",   32,   "diverse"),
    ("ER",   32,   "local"),
    ("ER",  128,   "diverse"),
    ("ER",  128,   "local"),
    ("ER", 1024,   "diverse"),
    ("ER", 1024,   "local"),
    ("SB",   32,   "diverse"),
    ("SB",   32,   "local"),
    ("SB",  128,   "diverse"),
    ("SB",  128,   "local"),
    ("SB", 1024,   "diverse"),
    ("SB", 1024,   "local"),
]
N_ARMS   = 100
P_ER     = 0.20
TRIALS   = 5
SEED     = 0
DTYPE    = np.float32

KAPPA_ARGS = dict(
    n_restarts=n_restarts, max_iter=200, step0=1.0, backtrack=0.5,
    dtype=DTYPE, minibatch=None, verbose=False
)

SAVE_DF  = Path("./figures/kappa_min_grid.csv")
SAVE_TEX = Path("./figures/kappa_min_grid.tex")

rng = np.random.default_rng(SEED)
rows = []

for index, (graph, n, mode) in enumerate(COMBOS):
    print(f"{index}/{len(COMBOS)}")
    if graph == "ER":
        L = generate_graph_laplacian(n, p=P_ER)
    else:  # SBM homogeneous with homophily
        L = generate_graph_laplacian(n, setting="homo")

    kappas, comms, min_eigens = [], [], []
    for t in range(TRIALS):
        if mode == "diverse":
            arms = generate_diverse_arms(n_arms=N_ARMS, n=n, p=P_ER)
        else:
            arms = generate_arms(L, k=2 * n, n_arms=N_ARMS)

        s_true = rng.normal(size=n)
        s_true -= s_true.mean()
        s_true /= (np.std(s_true) + 1e-12)

        try:
            comms.append(float(avg_commutator_norm(arms)))
        except Exception:
            comms.append(np.nan)

        try:
            kappa_hat, _ = kappa0_cone_fast(arms, s=s_true, **KAPPA_ARGS)
            kappas.append(float(kappa_hat))
            min_eig = min_eig_gram(arms)
            min_eigens.append(float(min_eig))
        except Exception:
            kappas.append(np.nan)

    rows.append({
        "graph": graph,
        "n": n,
        "mode": mode,
        "p": P_ER if graph == "ER" else np.nan,
        "kappa_mean": np.nanmean(kappas),
        "kappa_std":  np.nanstd(kappas),
        "min_eigen": np.mean(min_eigens)
    })

df = pd.DataFrame(rows).sort_values(["graph", "n", "mode"]).reset_index(drop=True)
SAVE_DF.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SAVE_DF, index=False)

# ---------- formatting helpers ----------
def format_kappa(mean, std, precision_plain=3, precision_sci=2):
    """
    Pretty (mean ± std) formatter:
      - plain decimals if magnitude is in [1e-3, 1e+3],
      - scientific otherwise, as (a ± b) × 10^{k}.
    """
    if np.isnan(mean) or np.isnan(std):
        return "--"
    amean = abs(mean)
    if (amean >= 1e-3) and (amean < 1e3):
        return rf"${mean:.{precision_plain}f} \; (\pm {std:.{precision_plain}f})$"
    exp = int(np.floor(np.log10(amean))) if amean > 0 else 0
    scale = 10.0 ** exp
    mant_mean = mean / scale
    mant_std  = std  / scale
    return rf"$({mant_mean:.{precision_sci}f} \pm {mant_std:.{precision_sci}f})\times 10^{{{exp}}}$"

# ---------- build df_fmt with formatted kappa strings ----------
df_fmt = df.copy()
df_fmt["kappa"] = [
    format_kappa(m, s) for m, s in zip(df["kappa_mean"], df["kappa_std"])
]

# ---------- pivot to wide format (rows = graph+mode, cols = n) ----------
df_wide = (
    df_fmt.pivot(index=["graph", "mode"], columns="n", values="kappa")
          .reset_index()
          .sort_values(["graph", "mode"])
)

# ensure we render columns in sorted |V| order
n_cols = sorted(df["n"].unique().tolist())

# ---------- build LaTeX table manually to match target style ----------
header = r"""
\begin{table}[h!]
\centering
\caption{Empirical lower bounds $\hat{\kappa}_0$ (mean $\pm$ std) across graph families and arm regimes. 
See main text for detailed description of graph models and arm generation procedures.}
\begin{tabular}{ll""" + "c" * len(n_cols) + r"""}
\toprule
Graph & Arm regime """ + "".join([f" & $|V|={n}$" for n in n_cols]) + r""" \\
\midrule
"""

rows_latex = []
for g in df_wide["graph"].unique():
    block = df_wide[df_wide["graph"] == g]
    for _, row in block.iterrows():
        line = f"{g} & {row['mode'].capitalize()}" + "".join(
            f" & {row.get(n, '--')}" for n in n_cols
        ) + r" \\"
        rows_latex.append(line)
    rows_latex.append(r"\midrule")

footer = r"""\bottomrule
\end{tabular}
\end{table}
"""

latex_full = "\n".join([header] + rows_latex[:-1] + [footer])  # drop last midrule

with open(SAVE_TEX, "w") as f:
    f.write(latex_full)

print(f"\nSaved CSV -> {SAVE_DF}\nSaved LaTeX -> {SAVE_TEX}")