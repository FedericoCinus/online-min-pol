import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
import seaborn as sns
import time

import sys
sys.path.append("./src")
from full_pipeline import run_experiment_and_plot
from viz import *

only_plot = False
debug     = False

trials   = 100        # number of Monte Carlo runs
lam      = 0.1       # ridge regularization λ
delta    = 1e-3      # confidence parameter
T        = 10_000    # total horizon
T1       = int(np.sqrt(T))   # Stage-1 budget
NAMES  = ["REAL_florentine_families_graph", "REAL_karate_club_graph", "REAL_davis_southern_women_graph", "REAL_les_miserables_graph", ]


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

# Precompute all combinations so we can track progress
total = len(NAMES)
start = time.perf_counter()

# --- experiment setup summary ---
print("-"*25)
print("\nStarting experiments with:")
print(f"  trials   = {trials}")
print(f"  lam      = {lam}")
print(f"  delta    = {delta}")
print(f"  T        = {T}")
print(f"  T1       = {T1}")
print()

for idx, name in enumerate(NAMES, start=1):
    
    G = nx.florentine_families_graph()
    if name == "REAL_florentine_families_graph":
        n = nx.florentine_families_graph().number_of_nodes()
    elif name == "REAL_karate_club_graph":
        n = nx.karate_club_graph().number_of_nodes()
    elif name == "REAL_davis_southern_women_graph":
        n = nx.davis_southern_women_graph().number_of_nodes()
    elif name == "REAL_les_miserables_graph":
        n = nx.les_miserables_graph().number_of_nodes()
    else:
        raise Exception(f"{name}")
    
    for n_arms in [10, 1_000]:
        for sigma in [1., .01]:
            n_e = int(2 * n)
            print(f"  name   = {name}")
            save_to = f"./figures/opdmin_vs_oful_{name}_sigma{int(100*sigma)}_arms{n_arms}_{n_e}.pdf"

            # --- run experiment ---
            run_experiment_and_plot(
                name=name, T=T, T1=T1, n_e=n_e, trials=trials,
                sigma=sigma, lam=lam, delta=delta,
                save_path=save_to, debug=debug,
                only_plot=only_plot,
                benchmark={"OUR"},
                skip_plot=True,
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
                f"current: n={n}, name={name}, sigma={sigma}, n_arms={n_arms}",
                flush=True
            )

print("\nFinished ✅")
print("\n", "-"*25)





# =============================================================
# ========================== PLOT =============================

network2paths = {
    "REAL_florentine_families_graph_arms_1000": [
        "./figures/opdmin_vs_oful_REAL_florentine_families_graph_sigma1_arms1000_30",
        "./figures/opdmin_vs_oful_REAL_florentine_families_graph_sigma100_arms1000_30",
    ],
    "REAL_davis_southern_women_graph_arms_1000": [
        "./figures/opdmin_vs_oful_REAL_davis_southern_women_graph_sigma1_arms1000_64",
        "./figures/opdmin_vs_oful_REAL_davis_southern_women_graph_sigma100_arms1000_64",
    ],
    "REAL_karate_club_graph_arms_1000": [
        "./figures/opdmin_vs_oful_REAL_karate_club_graph_sigma1_arms1000_68",
        "./figures/opdmin_vs_oful_REAL_karate_club_graph_sigma100_arms1000_68",
    ],
    "REAL_les_miserables_graph_arms_1000": [
        "./figures/opdmin_vs_oful_REAL_les_miserables_graph_sigma1_arms1000_154",
        "./figures/opdmin_vs_oful_REAL_les_miserables_graph_sigma100_arms1000_154"
    ],
    "REAL_florentine_families_graph_arms_10": [
        "./figures/opdmin_vs_oful_REAL_florentine_families_graph_sigma1_arms10_30",
        "./figures/opdmin_vs_oful_REAL_florentine_families_graph_sigma100_arms10_30",
    ],
    "REAL_davis_southern_women_graph_arms_10": [
        "./figures/opdmin_vs_oful_REAL_davis_southern_women_graph_sigma1_arms10_64",
        "./figures/opdmin_vs_oful_REAL_davis_southern_women_graph_sigma100_arms10_64",
    ],
    "REAL_karate_club_graph_arms_10": [
        "./figures/opdmin_vs_oful_REAL_karate_club_graph_sigma1_arms10_68",
        "./figures/opdmin_vs_oful_REAL_karate_club_graph_sigma100_arms10_68",
    ],
    "REAL_les_miserables_graph_arms_10": [
        "./figures/opdmin_vs_oful_REAL_les_miserables_graph_sigma1_arms10_154",
        "./figures/opdmin_vs_oful_REAL_les_miserables_graph_sigma100_arms10_154",
    ]

}

for network, PATHS in network2paths.items():

    # regex to capture sigma and arms
    pattern = re.compile(r"sigma(\d+)_arms(\d+)")

    # choose a color palette with enough distinct colors
    palette = sns.color_palette("tab10", len(PATHS))

    apply_style(font_size=9, line_width=2, use_tex=True)
    fig, ax = plt.subplots()

    for i, path in enumerate(PATHS):
        m = pattern.search(path)
        if not m:
            label = path
        else:
            sigma, arms = m.groups()
            # LaTeX style label
            label = fr"$\sigma={int(sigma)/100}$"

        # call your existing plot_data but suppress saving/showing,
        # pass ax, and override color + label
        ax = plot_data(
            n=None,
            orig_save_path=path,
            show=False,
            save=False,
            ax=ax,
            finalize=False,
        )

        # overwrite last line’s properties to force custom color/label
        line = ax.get_lines()[-1]
        line.set_color(palette[i])
        line.set_label(label)
        # match shaded area color too
        coll = ax.collections[-1]  # the PolyCollection from fill_between
        coll.set_facecolor(palette[i])
        coll.set_alpha(cfg["styles"]["ci_alpha"])

    # finalize once
    plt.title(fr'{network.split("arms")[0].replace("_", " ").replace("graph", "").replace("REAL", "").strip().capitalize()},    $|\mathcal{{X}}|={arms}$')
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(pad=0.2)
    # Save to file
    fig.savefig(f'./figures/{network}.pdf', bbox_inches="tight", pad_inches=0.01)
    plt.show()