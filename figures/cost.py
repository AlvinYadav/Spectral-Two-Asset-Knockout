"""Figures 6.2 and 6.3: cost at matched accuracy (6.2, the costs of Table 6.3 as bars; 6.3,
price error against cost).

Reads validation/artifacts/paper1_cost2_report.json (spectral costs and the matched
finite-difference grid), paper1_cost2_memo.json (Mathieu solve with the root memo on),
paper1_cost2_workprec.json (one finite-difference solve per grid) and the Monte Carlo
records validation/<contract>.toml. Writes p1_cost_tasks.pdf and p1_cost_accuracy.pdf to
PAPER_FIGURES_DIR.
"""
from __future__ import annotations

import json
import math
import sys
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from twoasset.solvers._optimal import ARTIFACTS_DIR, PAPER_FIGURES_DIR  # noqa: E402

ORDER = ["default", "tight_barrier", "mahalanobis", "dnt", "spread", "worst_of"]
TITLE = {"default": "Euclidean basket", "tight_barrier": "Euclidean, tight",
         "mahalanobis": "Mahalanobis basket", "dnt": "Double no-touch",
         "spread": "Spread call", "worst_of": "Worst-of call"}
GRID_SYMBOL = {"default": "m", "tight_barrier": "m", "mahalanobis": "m",
               "dnt": "N", "spread": "N", "worst_of": "N"}

# Colour follows the method (categorical slots 1-3, validated all-pairs); the marker
# shape repeats it, so no identity rests on colour alone.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#d9d8d4", "#eeede9"


def _json(name):
    return json.loads((Path(ARTIFACTS_DIR) / name).read_text())


def load():
    rep = _json("paper1_cost2_report.json")["rows"]
    memo = _json("paper1_cost2_memo.json")["rows"]
    wp = _json("paper1_cost2_workprec.json")["rows"]
    rows = {}
    for c in ORDER:
        r, sp = rep[c], rep[c]["spectral"]
        fd = r["fd_matched_single_grid"]
        mc = tomllib.loads((ROOT / "validation" / f"{c}.toml").read_text())["mc"]
        n_runs = len(mc["seeds"])
        row = dict(
            spec_scratch=sp["cold_s"] + sp["fresh_price_greeks_s"],
            spec_err=abs(sp["rel_err"]["price"]),
            spec_spot=sp["fresh_price_greeks_s"],
            spec_payoff=sp["new_payoff_on_basis_s"],
            spec_market=sp["market_greeks_s"],
            fd_level=fd["level"],
            fd_scratch=fd["price_and_spot_greeks_s"],
            fd_err=abs(fd["rel_err"]["price"]) if fd["rel_err"] else None,
            fd_spot=r["fd_evaluation"]["fresh_price_greeks"]["median"],
            fd_payoff=fd["new_payoff_lu_reused_s"],
            fd_market=fd["market_greeks_s"],
            ref_e=r["reference_e_rel"]["price"],
            # one run of 2e6 paths: its wall time and standard error, relative to the price
            mc_s=mc["wall_s"] / n_runs,
            mc_err=mc["sem"] * math.sqrt(n_runs) / abs(mc["mean"]),
            wp=sorted((p["seconds"]["median"], abs(p["rel_err"]["price"]), p["level"])
                      for p in wp[c]["levels"]),
        )
        q = r.get("spectral_refined_quadrature")
        if q:
            row.update(refined_scratch=q["cold_s"] + sp["fresh_price_greeks_s"],
                       refined_err=abs(q["rel_err"]["price"]))
        # the matched grid on the work-precision curve, timed in the same session as it
        on_curve = [s for s, _e, lv in row["wp"] if lv == row["fd_level"]]
        row["fd_scratch_wp"] = on_curve[0] if on_curve else row["fd_scratch"]
        if c in memo:
            row.update(memo_scratch=memo[c]["price_and_spot_greeks_from_scratch"]["median"],
                       memo_market=memo[c]["market_greeks_memo"]["median"])
        rows[c] = row
    return rows


def _style():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 8, "axes.linewidth": 0.6,
                         "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "figure.dpi": 300,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "hatch.color": "white", "hatch.linewidth": 0.6})


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(which="major", color=GRID, lw=0.4, zorder=0)


# ═══════════════════════════════════════════════════════════════════════
#  Figure 6.2: the costs of Table 6.3 as grouped bars
# ═══════════════════════════════════════════════════════════════════════

SHORT = {"default": "Euclidean\nbasket", "tight_barrier": "Euclidean,\ntight",
         "mahalanobis": "Mahalanobis\nbasket", "dnt": "Double\nno-touch",
         "spread": "Spread\ncall", "worst_of": "Worst-of\ncall"}
MEMO_HATCH = "////"


def _fmt(t):
    """Seconds as a bar label: two significant digits below 10, whole seconds above."""
    return f"{t:#.2g} s" if t < 10 else f"{t:.0f} s"


def _bars(ax, rows, series, title):
    """Grouped bars per contract on a linear axis, each bar labelled with its value, so
    that bars too short to see still carry their number. A series without a value for a
    contract (the memo outside the Mathieu contracts) leaves its slot empty."""
    n = len(series)
    width = 0.84 / n
    x = list(range(len(ORDER)))
    top = max(rows[c][k] for k, *_ in series for c in ORDER if rows[c].get(k) is not None)
    pos = {}                      # bars of each contract centred on the series it has
    for xi, c in zip(x, ORDER):
        have = [i for i, (k, *_r) in enumerate(series) if rows[c].get(k) is not None]
        for j, i in enumerate(have):
            pos[(c, i)] = xi + (j - (len(have) - 1) / 2) * width
    for i, (key, label, color, hatch) in enumerate(series):
        pts = [(pos[(c, i)], rows[c][key]) for c in ORDER if (c, i) in pos]
        xs, vs = zip(*pts)
        ax.bar(xs, vs, width=width * 0.9, color=color, edgecolor="white", linewidth=0.6,
               hatch=hatch, zorder=3, label=label)
        for xp, v in pts:
            ax.text(xp, v + 0.015 * top, _fmt(v), ha="center", va="bottom", rotation=90,
                    fontsize=6.4, color=INK2)
    ax.set_ylim(0, 1.6 * top)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in ORDER], fontsize=7.5, color=INK)
    ax.tick_params(axis="y", labelsize=7, length=2)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("seconds", fontsize=7.5, color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=7, ncol=n, handlelength=1.3,
              columnspacing=1.2)


def tasks_figure(rows):
    spectral = ("spec_{}", "spectral, roots scanned", BLUE, None)
    memo = ("memo_{}", "spectral, roots from the memo", BLUE, MEMO_HATCH)
    fd = ("fd_{}", "finite differences, matched grid", ORANGE, None)

    def ser(task, *items):
        return [(k.format(task), lab, col, h) for k, lab, col, h in items]
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 4.6))
    _bars(axes[0], rows, ser("scratch", spectral, memo, fd),
          "(a) Price and six spot Greeks, from scratch")
    _bars(axes[1], rows, ser("payoff", ("spec_{}", "spectral (projection)", BLUE, None),
                             ("fd_{}", "finite differences (time steps, LU reused)", ORANGE, None)),
          "(b) A new payoff on the same barrier and market")
    _bars(axes[2], rows, ser("market", spectral, memo, fd), "(c) The three market Greeks")
    fig.tight_layout(h_pad=1.0)
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Figure 6.3: price error against cost, from scratch
# ═══════════════════════════════════════════════════════════════════════

XLIM, YLIM = (8e-3, 6e2), (1e-7, 3e-2)


def accuracy_figure(rows):
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.15), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, ORDER):
        row = rows[c]
        ax.axhspan(YLIM[0], row["ref_e"], color=BAND, lw=0, zorder=0)
        ax.axhline(row["spec_err"], color=BLUE, lw=0.6, ls=(0, (2, 2)), zorder=1)
        t, e, lv = zip(*row["wp"])
        ax.plot(t, e, "-s", color=ORANGE, lw=1.0, ms=2.8, mec="white", mew=0.3, zorder=2)
        sym = GRID_SYMBOL[c]
        for i in (0, len(lv) - 1):
            if lv[i] == row["fd_level"]:
                continue                         # labelled as the matched grid below
            ax.annotate(f"${sym}={lv[i]}$", (t[i], e[i]), textcoords="offset points",
                        xytext=(3, 3), fontsize=6, color=INK2, ha="left")
        if row["fd_err"] is not None:
            ax.plot(row["fd_scratch_wp"], row["fd_err"], "s", ms=6, color=ORANGE, mec=INK,
                    mew=0.6, zorder=4)
            ax.annotate(f"${sym}={row['fd_level']}$", (row["fd_scratch_wp"], row["fd_err"]),
                        textcoords="offset points", xytext=(4, -9), fontsize=6, color=INK)
        ax.plot(row["spec_scratch"], row["spec_err"], "o", ms=6, color=BLUE, mec="white",
                mew=0.6, zorder=5)
        if "memo_scratch" in row:
            ax.plot(row["memo_scratch"], row["spec_err"], "o", ms=6, mfc="white", mec=BLUE,
                    mew=1.2, zorder=5)
        if "refined_scratch" in row:
            ax.plot(row["refined_scratch"], row["refined_err"], "^", ms=5.5, color=BLUE,
                    mec="white", mew=0.5, zorder=5)
        ax.plot(row["mc_s"], row["mc_err"], "D", ms=4.5, color=AQUA, mec="white", mew=0.5,
                zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_title(TITLE[c], loc="left", fontsize=8, color=INK)
        ax.tick_params(length=2)
        _clean(ax)
    fig.supxlabel("seconds for the price and its six spot Greeks, from scratch", fontsize=7,
                  color=INK2, y=0.085)
    for ax in axes[:, 0]:
        ax.set_ylabel("relative price error", fontsize=7, color=INK2)
    handles = [Line2D([], [], color=ORANGE, marker="s", ms=2.8, lw=1.0, mec="white",
                      label="finite differences, one solve per grid"),
               Line2D([], [], ls="", marker="s", ms=6, color=ORANGE, mec=INK, mew=0.6,
                      label="matched grid"),
               Line2D([], [], ls="", marker="o", ms=6, color=BLUE, mec="white",
                      label="spectral, roots scanned"),
               Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec=BLUE, mew=1.2,
                      label="spectral, roots from the memo"),
               Line2D([], [], ls="", marker="^", ms=5.5, color=BLUE, mec="white",
                      label="spectral, quadrature doubled"),
               Line2D([], [], ls="", marker="D", ms=4.5, color=AQUA, mec="white",
                      label=r"Monte Carlo, one run of $2\times10^6$ paths"),
               Line2D([], [], color=BLUE, lw=0.6, ls=(0, (2, 2)),
                      label="spectral price error"),
               Patch(color=BAND, label="below the extrapolation step $e$")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.03), handletextpad=0.4, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.8, w_pad=0.6)
    return fig


def main():
    rows = load()
    _style()
    out = Path(PAPER_FIGURES_DIR)
    for name, fig in (("p1_cost_tasks", tasks_figure(rows)),
                      ("p1_cost_accuracy", accuracy_figure(rows))):
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200)
        print(f"wrote {out / name}.pdf")


if __name__ == "__main__":
    main()
