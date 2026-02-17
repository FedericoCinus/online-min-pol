import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import StrMethodFormatter, MaxNLocator
from pathlib import Path
import re
import seaborn as sns




# Okabe–Ito palette (color-blind safe), muted for print
cfg = {
    "colors": {
        "ofu":    "#0072B2",  # blue
        "opd":    "#E69F00",  # orange
        "oracle": "#009E73",  # green
        "ref":    "#6A6A6A",  # neutral gray for theory baselines
        "offline": "#D55E00",  # vermilion for offline baseline
    },
    "styles": {
        # line styles
        "ofu_linestyle":     "-",
        "opd_linestyle":     "-",
        "oracle_linestyle":  "--",  # oracle is dashed to signal idealized baseline
        "ref_linestyle":     ":",
        "offline_linestyle": "-.",

        # thickness & transparency
        "linewidth": 2.0,
        "ci_alpha": 0.18,          # shaded 95% CI
        "ref_alpha": 0.8,

        # labels
        "xlabel": "Iterations",
        "ylabelR": "Cumulative Regret",
        "ylabelM": "Minimum Objective Value",
    },
}



# =============================================================
# ========================== PLOTS ============================
# =============================================================

def apply_style(font_size=6.5, line_width=1.0, use_tex=True):
    import matplotlib as mpl
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.figsize": (3.0, 1.9),   # compact: 2-per-row (fits ~7 in text width)
        "savefig.dpi": 300, "figure.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": line_width,
        "font.size": font_size,
        "axes.titlesize": font_size, "axes.labelsize": font_size,
        "xtick.labelsize": font_size, "ytick.labelsize": font_size,
        # Legend tweaks for compactness
        "legend.fontsize": font_size * 0.7,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.2,
        "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.2,
    })
    if use_tex:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
def format_ticks_thousands(ax, prefix):
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    if prefix == "R":
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # Optional: minor ticks for a more “technical” look
    ax.minorticks_off()  # or set to 'on' if you add minor locators


def _thousands(ax):
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

def _mean_ci(arr):
    """Return mean and 95% CI half-width across runs (axis=0)."""
    runs = arr.shape[0]
    m = arr.mean(0)
    se = arr.std(0, ddof=1) / np.sqrt(runs)
    return m, 1.96 * se

from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(
    n,
    orig_save_path: str,
    show: bool = False,
    save: bool = True,
    *,
    ax: plt.Axes | None = None,   # ← new: draw onto this axes if given
    finalize: bool = True,        # ← new: add legend/save/show only when True
    prefix: str = "R"
):
    # 1) Resolve output locations
    save_path = orig_save_path.format(n=n) if "{n}" in orig_save_path else orig_save_path
    save_path = Path(save_path)
    data_dir = save_path.with_suffix("")

    # 2) Load only if they exist
    def load_if_exists(path): return np.load(path) if path.exists() else None
    R_opd_     = load_if_exists(data_dir / f"{prefix}_opd.npy")
    R_ofu_     = load_if_exists(data_dir / f"{prefix}_ofu.npy")
    R_oracle_  = load_if_exists(data_dir / f"{prefix}_oracle.npy")
    R_offline_ = load_if_exists(data_dir / f"{prefix}_offline.npy")
    meta = np.load(data_dir / "meta.npz", allow_pickle=False) if (data_dir / "meta.npz").exists() else None
    labs = json.load(open(data_dir / "labels.json")) if (data_dir / "labels.json").exists() else {}

    if all(r is None for r in [R_opd_, R_ofu_, R_oracle_, R_offline_]):
        print(f"No data found in {data_dir}, skipping plot.")
        return ax  # nothing added

    x = np.arange(int(meta["T"])) if meta is not None else None
    def safe_mean_ci(R): return _mean_ci(R) if R is not None else (None, None)
    m_opd, e_opd       = safe_mean_ci(R_opd_)
    m_ofu, e_ofu       = safe_mean_ci(R_ofu_)
    m_oracle, e_oracle = safe_mean_ci(R_oracle_)
    m_offline, e_offline = safe_mean_ci(R_offline_)

    # 3) Style + get axes
    created_ax = ax is None
    if created_ax:
        apply_style(font_size=9, line_width=2, use_tex=True)
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # 4) Plot (adds lines to given ax)
    if m_oracle is not None:
        ax.plot(x, m_oracle, cfg["styles"]["oracle_linestyle"],
                color=cfg["colors"]["oracle"], label=labs.get("lab_oracle", "Oracle"))
        ax.fill_between(x, m_oracle - e_oracle, m_oracle + e_oracle,
                        color=cfg["colors"]["oracle"], alpha=cfg["styles"]["ci_alpha"], linewidth=0)

    if m_opd is not None:
        ax.plot(x, m_opd, cfg["styles"]["opd_linestyle"],
                color=cfg["colors"]["opd"], label=labs.get("lab_opd", "OPD"))
        ax.fill_between(x, m_opd - e_opd, m_opd + e_opd,
                        color=cfg["colors"]["opd"], alpha=cfg["styles"]["ci_alpha"], linewidth=0)

    if m_ofu is not None:
        ax.plot(x, m_ofu, cfg["styles"]["ofu_linestyle"],
                color=cfg["colors"]["ofu"], label=labs.get("lab_ofu", "OFU"))
        ax.fill_between(x, m_ofu - e_ofu, m_ofu + e_ofu,
                        color=cfg["colors"]["ofu"], alpha=cfg["styles"]["ci_alpha"], linewidth=0)
    if m_offline is not None:
        ax.plot(x, m_offline, cfg["styles"]["ref_linestyle"],
                color=cfg["colors"]["ref"], label=labs.get("lab_offline", "Offline"))
        ax.fill_between(x, m_offline - e_offline, m_offline + e_offline,
                        color=cfg["colors"]["ref"], alpha=cfg["styles"]["ci_alpha"], linewidth=0)

    ax.set_xlabel(cfg["styles"]["xlabel"])
    ax.set_ylabel(cfg["styles"][f"ylabel{prefix}"])
    format_ticks_thousands(ax, prefix)
    sns.despine(ax=ax)

    # 5) Only once at the end
    if finalize:
        loc = "upper left" if prefix == "R" else "upper right"
        ax.legend(loc=loc) 
        plt.tight_layout(pad=0.2)
        if save:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax


def plot_scalability_curve(rows, OUT):
    if not rows:
        print("No scalability runs found.")
        return

    ns    = np.array([r["n"] for r in rows], dtype=int)
    m_opd = np.array([np.nan if r["m_opd"] is None else r["m_opd"] for r in rows], dtype=float)
    s_opd = np.array([0.0   if r["s_opd"] is None else r["s_opd"] for r in rows], dtype=float)

    have_ofu = any(r["m_ofu"] is not None for r in rows)
    if have_ofu:
        m_ofu = np.array([np.nan if r["m_ofu"] is None else r["m_ofu"] for r in rows], dtype=float)
        s_ofu = np.array([0.0   if r["s_ofu"] is None else r["s_ofu"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(4.0, 2.6), dpi=300)

    # Log–log scaling
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    # Format x-ticks as powers of 2: 2^k
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: rf"$2^{{{int(np.log2(x))}}}$" if x > 0 else "")
    )

    # Our method
    ax.plot(ns, m_opd, marker="o", lw=1.8, label="Our", color=cfg["colors"]["opd"])
    ax.fill_between(ns, m_opd - s_opd, m_opd + s_opd, alpha=0.2, lw=0)

    # OFUL baseline (if present)
    if have_ofu:
        ax.plot(ns, m_ofu, marker="s", lw=1.8, color=cfg["colors"]["ofu"], label="OFUL")
        ax.fill_between(ns, m_ofu - s_ofu, m_ofu + s_ofu, alpha=0.2, lw=0)

    ax.set_xlabel(r"Number of nodes $|V|$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {OUT}")



# =============================================================
# ========================== UTILS ============================
# =============================================================

ROOT = Path("./figures")

def parse_n(name: str):
    m = re.search(r"_n(\d+)_", name)
    return int(m.group(1)) if m else None

def load_meta(dirpath: Path):
    mp = dirpath / "meta.npz"
    if not mp.exists(): 
        return None
    try:
        m = np.load(mp, allow_pickle=False)
        def _stat(key):
            arr = m.get(key, None)
            if arr is None or arr.size == 0: 
                return None, None
            arr = np.asarray(arr).ravel()
            return float(np.mean(arr)), float(np.std(arr))
        m_opd, s_opd = _stat("t_opd")
        m_ofu, s_ofu = _stat("t_ofu")
        return dict(n=None, m_opd=m_opd, s_opd=s_opd, m_ofu=m_ofu, s_ofu=s_ofu)
    except Exception:
        return None

def collect_rows_scalability():
    rows = []
    for p in sorted(ROOT.glob("scalability*")):
        data_dir = p if (p / "meta.npz").exists() else p.with_suffix("")
        if not (data_dir / "meta.npz").exists():
            continue
        r = load_meta(data_dir)
        if r is None:
            continue
        r["n"] = parse_n(p.name) or parse_n(data_dir.name)
        if r["n"] is None:
            continue
        rows.append(r)
    return sorted(rows, key=lambda r: r["n"])