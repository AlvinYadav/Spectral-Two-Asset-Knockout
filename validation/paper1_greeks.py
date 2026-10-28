"""Greeks: the effect of the benchmark's bump size (Table 6.2) and the Greek surfaces
(Figure 6.1)."""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")   # headless: never open a GUI window

import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

from paper1_evidence import build
from twoasset.runlog import build_provenance
from twoasset.solvers._optimal import (ALL_CONFIGS,
                                       ARTIFACTS_DIR, PAPER_FIGURES_DIR)

OUT = Path(ARTIFACTS_DIR) / "paper1_greeks.json"


def spectral_bump(solver, bump_pct=1.0, time_bump_frac=0.01):
    """fdm_greeks.fdm_greeks's stencils, applied to the spectral price."""
    m = solver.market
    S1, S2 = m.S1_0, m.S2_0
    h1, h2 = S1 * bump_pct / 100.0, S2 * bump_pct / 100.0
    P = solver.price
    V0 = P(S1, S2)
    Vp1, Vm1 = P(S1 + h1, S2), P(S1 - h1, S2)
    Vp2, Vm2 = P(S1, S2 + h2), P(S1, S2 - h2)
    Vpp, Vpm = P(S1 + h1, S2 + h2), P(S1 + h1, S2 - h2)
    Vmp, Vmm = P(S1 - h1, S2 + h2), P(S1 - h1, S2 - h2)
    dt = time_bump_frac * m.T
    V_Tm = P(S1, S2, t=dt)          # constant coefficients: maturity T - dt
    return dict(
        price=V0,
        delta1=(Vp1 - Vm1) / (2 * h1),
        delta2=(Vp2 - Vm2) / (2 * h2),
        gamma1=(Vp1 - 2 * V0 + Vm1) / h1 ** 2,
        gamma2=(Vp2 - 2 * V0 + Vm2) / h2 ** 2,
        gamma12=(Vpp - Vpm - Vmp + Vmm) / (4 * h1 * h2),
        theta=-(V0 - V_Tm) / dt,
    )


def bump():
    rows = []
    for name in ALL_CONFIGS:
        _cfg, solver = build(name)
        solver.fit(); solver.project()
        analytic = {k: float(v) for k, v in solver.greeks().items()}
        sb = {k: float(v) for k, v in spectral_bump(solver).items()}
        fdm = tomllib.loads((ROOT / "validation" / f"{name}.toml").read_text())["fdm_greeks"]
        fdm = {k: float(fdm[k]) for k in sb}
        rows.append(dict(config=name, analytic=analytic, spectral_bump=sb, fdm_bump=fdm))
        print(f"[bump] {name}", flush=True)
        for k in ("delta1", "delta2", "gamma1", "gamma2", "gamma12", "theta"):
            print(f"    {k:8s} analytic {analytic[k]: .7g}  spec-bump {sb[k]: .7g}  "
                  f"fdm-bump {fdm[k]: .7g}", flush=True)
    return rows


def _clear_eval_caches(solver):
    """Release the evaluation memos between rings."""
    basis = solver.basis
    for ang in getattr(basis, "_backend_solver_cache", {}).values():
        cache = getattr(ang, "_bessel_cache", None)
        if cache is not None:
            cache.clear()
    sep = getattr(basis, "_sep_eval_cache", None)
    if sep is not None:
        sep.clear()


def surface(n_r=22, n_theta=60, r_frac=0.995):
    """Price and spot Greeks on a polar grid in log-return space, which follows the
    circular barrier so the plotted region has a smooth edge."""
    _cfg, solver = build("default")
    solver.fit(); solver.project()
    m, R = solver.market, solver.domain.R
    rr = r_frac * R * np.arange(1, n_r + 1) / n_r
    th = 2.0 * np.pi * np.arange(n_theta) / n_theta
    keys = ("price", "delta1", "delta2", "gamma1", "gamma2", "gamma12", "theta")
    F = {k: np.full((n_r + 1, n_theta), np.nan) for k in keys}
    t0 = time.perf_counter()
    g0 = solver.greeks(m.S1_0, m.S2_0)                 # centre, r = 0
    for k in keys:
        F[k][0, :] = g0[k]
    n = 1
    for i, r in enumerate(rr, start=1):
        for j, t in enumerate(th):
            g = solver.greeks(m.S1_0 * np.exp(r * np.cos(t)), m.S2_0 * np.exp(r * np.sin(t)))
            for k in keys:
                F[k][i, j] = g[k]
            n += 1
        _clear_eval_caches(solver)          # keep memory bounded between rings
    wall = time.perf_counter() - t0
    print(f"[surface] {n} spots in {wall:.1f}s ({1e3 * wall / n:.1f} ms per spot)", flush=True)
    return dict(grid="polar", n_r=n_r, n_theta=n_theta, r_frac=r_frac, n_spots=n,
                wall_s=wall, R=R, r=[0.0] + rr.tolist(), theta=th.tolist(),
                fields={k: v.tolist() for k, v in F.items()})


def plot_surface(sd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 8.5, "axes.linewidth": 0.7,
                         "axes.titlesize": 9, "figure.dpi": 300})
    r = np.array(sd["r"])
    th = np.array(sd["theta"] + [sd["theta"][0] + 2.0 * np.pi])   # close the ring
    Rg, Tg = np.meshgrid(r, th, indexing="ij")
    S1 = 100.0 * np.exp(Rg * np.cos(Tg))
    S2 = 100.0 * np.exp(Rg * np.sin(Tg))
    R = sd["R"]
    tb = np.linspace(0, 2 * np.pi, 400)
    bS1, bS2 = 100.0 * np.exp(R * np.cos(tb)), 100.0 * np.exp(R * np.sin(tb))
    panels = [("price", r"(a) Price $V$", "viridis", False, 1.0),
              ("delta1", r"(b) $\Delta_1$", "RdBu_r", True, 1.0),
              ("delta2", r"(c) $\Delta_2$", "RdBu_r", True, 1.0),
              ("gamma1", r"(d) $\Gamma_{11}$  ($\times 10^{-2}$)", "RdBu_r", True, 1e2),
              ("gamma12", r"(e) $\Gamma_{12}$  ($\times 10^{-2}$)", "RdBu_r", True, 1e2),
              ("theta", r"(f) $\Theta$", "RdBu_r", True, 1.0)]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7))
    for ax, (k, title, cmap, diverging, scale) in zip(axes.ravel(), panels):
        Z = scale * np.array(sd["fields"][k], dtype=float)
        Z = np.concatenate([Z, Z[:, :1]], axis=1)                 # close the ring
        norm = None
        if diverging:
            lim = np.nanmax(np.abs(Z))
            norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        pc = ax.pcolormesh(S1, S2, Z, cmap=cmap, norm=norm, shading="gouraud",
                           rasterized=True)
        ax.plot(bS1, bS2, color="#111827", lw=0.9)
        ax.plot(100, 100, "o", ms=2.8, color="#111827")
        ax.set_aspect("equal")
        ax.set_xlim(56, 172)
        ax.set_ylim(56, 172)
        ax.set_xticks([60, 100, 140])
        ax.set_yticks([60, 100, 140])
        ax.tick_params(labelsize=7, length=2)
        ax.set_title(title, pad=3)
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6.5)
    for ax in axes.ravel():
        ax.set_xlabel(r"$S_1$", labelpad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$S_2$", labelpad=1)
    fig.tight_layout(w_pad=0.6, h_pad=0.4)
    out = Path(PAPER_FIGURES_DIR)
    fig.savefig(out / "p1_greek_surfaces.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out / "p1_greek_surfaces.png", bbox_inches="tight", dpi=200)
    print(f"wrote {out / 'p1_greek_surfaces.pdf'}")


def main(argv):
    parts = set(argv) or {"bump", "surface"}
    # "plot" alone redraws the figure from the saved data without recomputing.
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "bump" in parts:
        data["bump"] = bump()
    if "surface" in parts:
        data["surface"] = surface()
    if parts & {"bump", "surface"}:     # "plot" alone leaves the artefact as it is
        data["provenance"] = build_provenance()
        OUT.write_text(json.dumps(data, indent=2))
        print(f"wrote {OUT}")
    if "plot" in parts or "surface" in parts:
        plot_surface(data["surface"])


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(0)     # outputs are written; skip tearing down large caches
