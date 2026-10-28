"""Figure 6.2: the costs of Table 6.3, read from
validation/artifacts/paper1_evidence.json and validation/<contract>.toml."""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twoasset.solvers._optimal import ARTIFACTS_DIR, PAPER_FIGURES_DIR

LABELS = {
    "default": "Euclidean basket, $R=0.5$",
    "tight_barrier": "Euclidean basket, $R=0.3$",
    "mahalanobis": "Mahalanobis basket",
    "dnt": "Double no-touch",
    "spread": "Spread call",
    "worst_of": "Worst-of down-and-out call",
}
ORDER = ["default", "tight_barrier", "mahalanobis", "dnt", "spread", "worst_of"]

# Colour follows the method.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FD_SOLVES, FD_SPOT_SOLVES = 16, 10   # finite-difference solves: all Greeks, spot Greeks
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def load():
    ev = json.loads((Path(ARTIFACTS_DIR) / "paper1_evidence.json").read_text())
    cost = {r["config"]: r for r in ev["cost"]}
    rows = {}
    for c in ORDER:
        v = tomllib.loads((ROOT / "validation" / f"{c}.toml").read_text())
        r = cost[c]
        fd_set = v["fdm_greeks"]["wall_s"]
        rows[c] = dict(
            spec_price=r["cold_total_s"] + r["price_fresh_median_s"],
            fdm_price=r["fdm_wall_s"],
            mc_price=v["mc"]["wall_s"] / len(v["mc"]["seeds"]),
            spec_spot=r["cold_total_s"] + r["greeks_fresh_median_s"],
            fdm_spot=fd_set * FD_SPOT_SOLVES / FD_SOLVES,
            spec_market=r["greeks_full_s"],
            fdm_market=fd_set * (FD_SOLVES - FD_SPOT_SOLVES) / FD_SOLVES,
        )
    return rows


SHORT = {
    "default": "Euclidean\nbasket",
    "tight_barrier": "Euclidean,\ntight",
    "mahalanobis": "Mahalanobis\nbasket",
    "dnt": "Double\nno-touch",
    "spread": "Spread\ncall",
    "worst_of": "Worst-of\ncall",
}


def _fmt(t):
    """Seconds -> compact label, always in seconds (two significant digits)."""
    return f"{t:.2g} s" if t < 10 else f"{t:.0f} s"


def _bars(ax, series, title):
    """Grouped bars per contract on a linear axis in seconds, each bar labelled with
    its value, so that bars too short to see still carry their number."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(ORDER))
    top = max(ROWS[c][key] for key, *_ in series for c in ORDER)
    for i, (key, label, color, hatch) in enumerate(series):
        vals = np.array([ROWS[c][key] for c in ORDER])
        pos = x + (i - (n - 1) / 2) * width
        ax.bar(pos, vals, width=width * 0.92, color=color, edgecolor="white",
               linewidth=0.6, hatch=hatch, label=label, zorder=3)
        for xp, v in zip(pos, vals):
            ax.text(xp, v + 0.015 * top, _fmt(v), ha="center", va="bottom", rotation=90,
                    fontsize=6.6, color=INK2)
    ax.set_ylim(0, 1.62 * top)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.grid(axis="y", which="major", color=GRID, lw=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in ORDER], fontsize=8)
    ax.tick_params(axis="y", labelsize=8, length=2)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("wall-clock time (s)", fontsize=8.5, color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=7.8, ncol=3,
              handlelength=1.4, columnspacing=1.2)


def main():
    global ROWS
    ROWS = load()
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 9, "axes.linewidth": 0.6,
                         "axes.edgecolor": INK2, "xtick.color": INK,
                         "ytick.color": INK2, "figure.dpi": 300,
                         "hatch.color": "white", "hatch.linewidth": 0.6})
    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(6.6, 6.3))
    _bars(ax_a, [("spec_price", "spectral, from scratch", BLUE, None),
                 ("fdm_price", "finite differences, one solve", ORANGE, None),
                 ("mc_price", r"Monte Carlo, $2\times10^6$ paths", AQUA, None)],
          "(a) One price")
    _bars(ax_b, [("spec_spot", "spectral, from scratch", BLUE, None),
                 ("fdm_spot", "finite differences (10 solves)", ORANGE, None)],
          "(b) Price with the six spot Greeks, from scratch")
    _bars(ax_c, [("spec_market", "spectral (6 re-solves)", BLUE, None),
                 ("fdm_market", "finite differences (6 solves)", ORANGE, None)],
          "(c) The three market Greeks")
    fig.tight_layout(h_pad=1.4)
    out = Path(PAPER_FIGURES_DIR)
    fig.savefig(out / "p1_cost.pdf", bbox_inches="tight")
    fig.savefig(out / "p1_cost.png", bbox_inches="tight", dpi=200)
    print(json.dumps(ROWS, indent=1))
    print(f"wrote {out / 'p1_cost.pdf'}")


if __name__ == "__main__":
    main()
