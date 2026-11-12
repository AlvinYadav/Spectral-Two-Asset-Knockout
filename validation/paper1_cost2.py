"""Cost of comparable tasks, spectral against finite differences (Paper 1).

For every contract and method:
  (1) preprocessing   spectral basis (fit) | FD assembly and LU
  (2) per payoff      projection            | time stepping
  (3) a further spot price on the solved basis / stored grid solution
  (4) a further price with its six spot Greeks
  (5) a batch of spots (the 2,032-spot grid of Table 6.4)
  (6) the three market Greeks (spectral: four rebuilds and two projections;
      FD: six re-solves)
and, for the FD, every grid level, so that the cost at matched accuracy can be read
off against the references of paper1_zfdm.py (curved barriers, z-space FD) and of
its xgreeks section (rectangles, x-space FD).

Single-threaded: the thread variables below are set before NumPy is imported. Each
timed task has one discarded warm-up, then the median and interquartile range over
REPEATS runs (fresh-spot tasks: over the spots). The Mathieu root memo is off, so
every spectral fit scans its own roots. Nothing else may run during the timing.

The `matched` command times one FD solve on the coarsest grid that matches the
spectral price error (see the section below).

Run (repo root):  PYTHONPATH=. .venv/bin/python3 validation/paper1_cost2.py run [CONFIG ...]
                  PYTHONPATH=. .venv/bin/python3 validation/paper1_cost2.py matched [CONFIG ...]
                  PYTHONPATH=. .venv/bin/python3 validation/paper1_cost2.py report
Writes:           validation/artifacts/paper1_cost2.json (run, per contract, merged)
                  validation/artifacts/paper1_cost2_matched.json (matched)
                  validation/artifacts/paper1_cost2_report.json (report)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json                                                          # noqa: E402
import platform                                                      # noqa: E402
import subprocess                                                    # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
import tomllib                                                       # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402
import scipy                                                         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

import twoasset.bases.mathieu as mathieu_basis                       # noqa: E402
from paper1_evidence import (_NoRootMemo, _batch_price, _batch_spots,  # noqa: E402
                             _ring, build)
from twoasset.cli import assemble                                    # noqa: E402
from twoasset.runlog import build_provenance                         # noqa: E402
from twoasset.solvers._optimal import ALL_CONFIGS, ARTIFACTS_DIR, FOURIER_CONFIGS  # noqa: E402
from twoasset.solvers.fdm_grid_greeks import fdm_x_onesolve          # noqa: E402
from twoasset.solvers.fdm_z import _bumped, fdm_z_solve              # noqa: E402

ART = Path(ARTIFACTS_DIR)
OUT = ART / "paper1_cost2.json"
REPORT = ART / "paper1_cost2_report.json"
REPEATS = 5
CURVED = ("default", "tight_barrier", "mahalanobis")
Z_LEVELS = ((40, 125), (80, 250), (160, 500), (320, 1000))
# Gate Z6: at the paired step counts the time error is ~50x below the space error, so
# a quarter of the steps (still dt ∝ h, so the two-grid extrapolation stays second
# order) is the FD's cheaper work-precision line.
Z_LEVELS_FAST = ((40, 32), (80, 63), (160, 125), (320, 250))
X_LEVELS = ((160, 250), (320, 500), (640, 1000))
# The spectral counterpart: the one-factor study puts the remaining curved-barrier
# and double no-touch error in the quadrature, so doubling it is the spectral
# method's more accurate point.
SPECTRAL_REFINED = {
    "default": dict(numerical=dict(N_mu_quad=100, N_nu_quad=192)),
    "tight_barrier": dict(numerical=dict(N_mu_quad=100, N_nu_quad=192)),
    "mahalanobis": dict(numerical=dict(N_r_quad=80, N_theta_quad=192)),
    "dnt": dict(N_quad=480),
}
KEYS = ("price", "delta1", "delta2", "gamma1", "gamma2", "gamma12", "theta")
X_ORDERING = "MMD_AT_PLUS_A"         # fill-reducing LU ordering for the x-space timings
N_FRESH = 12


def _pc():
    return time.perf_counter()


def _stats(ts):
    ts = np.asarray(ts, dtype=float)
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return dict(median=float(med), iqr=float(q3 - q1), n=int(len(ts)),
                min=float(ts.min()), max=float(ts.max()))


def _repeat(fn, repeats=REPEATS):
    fn()                                            # discarded warm-up
    ts = []
    for _ in range(repeats):
        t0 = _pc()
        fn()
        ts.append(_pc() - t0)
    return _stats(ts)


def _hardware():
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                                  text=True).stdout.strip()
        except OSError:
            return None
    blas = None
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:                                # noqa: BLE001
        pass
    return dict(chip=sysctl("machdep.cpu.brand_string"),
                memory_bytes=sysctl("hw.memsize"), os=platform.platform(),
                python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__, blas=blas,
                threads={v: os.environ[v] for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                    "VECLIB_MAXIMUM_THREADS")})


# ═══════════════════════════════════════════════════════════════════════
#  spectral
# ═══════════════════════════════════════════════════════════════════════

def spectral_costs(name):
    fit, proj = [], []
    solver = None
    for k in range(REPEATS + 1):
        _c, s = build(name)
        t0 = _pc(); s.fit(); t1 = _pc(); s.project(); t2 = _pc()
        if k:
            fit.append(t1 - t0)
            proj.append(t2 - t1)
        solver = s
    m = solver.market
    solver.price()
    solver.greeks()
    out = dict(n_modes=int(solver.n_modes()), fit=_stats(fit), project=_stats(proj),
               cold=_stats(np.add(fit, proj)))
    ring_a = _ring(solver, 0.04, N_FRESH, 0.1)
    ring_b = _ring(solver, 0.07, N_FRESH, 0.37)
    tp, tg = [], []
    for a, b in ring_a:
        t0 = _pc(); solver.price(a, b); tp.append(_pc() - t0)
    for a, b in ring_b:
        t0 = _pc(); solver.greeks(a, b); tg.append(_pc() - t0)
    out["fresh_price"] = _stats(tp)
    out["fresh_price_greeks"] = _stats(tg)
    out["hot_price"] = _repeat(lambda: solver.price(m.S1_0, m.S2_0), 50)
    out["hot_price_greeks"] = _repeat(lambda: solver.greeks(m.S1_0, m.S2_0), 20)
    bS1, bS2 = _batch_spots(solver)
    st = _repeat(lambda: _batch_price(solver, bS1, bS2), 3)
    out["batch"] = dict(points=int(len(bS1)), total=st,
                        per_point_median=st["median"] / max(len(bS1), 1))
    reps = 2 if name in CURVED[:2] else REPEATS
    out["market_greeks"] = _repeat(
        lambda: solver._market_greeks(m.S1_0, m.S2_0, 0.0), reps)
    if name in CURVED[:2]:
        out["roots_cached"] = _roots_cached(name)
    if name in SPECTRAL_REFINED:
        out["refined"] = _refined(name)
    return out


def _refined(name):
    """Cold solve and values at the spot with the quadrature doubled."""
    spec = dict(SPECTRAL_REFINED[name])
    nq = spec.pop("N_quad", None)
    ts, s = [], None
    for k in range(3):
        _c, s = build(name, **spec)
        if nq is not None:
            s.basis.N_quad = nq
        t0 = _pc(); s.fit(); s.project(); t1 = _pc()
        if k:
            ts.append(t1 - t0)
    g = s.greeks()
    return dict(settings=SPECTRAL_REFINED[name], cold=_stats(ts),
                values={k: float(g[k]) for k in KEYS})


def _roots_cached(name):
    """Mathieu solve with the Stage 1 root memo serving the roots of the OTHER
    Euclidean contract on the same market (same Σ, other radius, hence the same
    mu_0 bits). Norms, the dedup Gram check,
    the projection and pricing are recomputed, as on every fit."""
    other = "tight_barrier" if name == "default" else "default"
    mathieu_basis._ROOT_CACHE = {}
    try:
        _c, s = build(other)
        s.fit()                                   # populates the memo (not timed)
        fit, proj, hits = [], [], []
        for k in range(REPEATS + 1):
            _c, s = build(name)
            t0 = _pc(); s.fit(); t1 = _pc(); s.project(); t2 = _pc()
            if k:
                fit.append(t1 - t0)
                proj.append(t2 - t1)
                hits.append(bool(s.basis._root_cache_hit))
        s.price()
        t0 = _pc(); s.greeks(); t_g = _pc() - t0
        out = dict(memo_source=other, all_hits=all(hits), fit=_stats(fit),
                   project=_stats(proj), cold_with_cached_roots=_stats(np.add(fit, proj)),
                   first_greeks_at_spot_s=t_g, price=float(s.price()))
        # The same contract refitted with its own roots in the memo (a recalibration
        # that leaves Σ and the barrier unchanged).
        mathieu_basis._ROOT_CACHE = {}
        _c, s = build(name)
        s.fit()                                   # populates the memo (not timed)
        fit, proj, hits = [], [], []
        for k in range(REPEATS + 1):
            _c, s = build(name)
            t0 = _pc(); s.fit(); t1 = _pc(); s.project(); t2 = _pc()
            if k:
                fit.append(t1 - t0)
                proj.append(t2 - t1)
                hits.append(bool(s.basis._root_cache_hit))
        out["same_contract"] = dict(all_hits=all(hits), fit=_stats(fit), project=_stats(proj),
                                    cold_with_cached_roots=_stats(np.add(fit, proj)),
                                    price=float(s.price()))
        return out
    finally:
        mathieu_basis._ROOT_CACHE = _NoRootMemo()


# ═══════════════════════════════════════════════════════════════════════
#  finite differences
# ═══════════════════════════════════════════════════════════════════════

def _contract(name):
    cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text())
    market, payoff, domain, _b, _ = assemble(cfg)
    return market, payoff, domain


def _fd_eval_costs(sol, market, ring_a, ring_b, bS1, bS2):
    t0 = _pc(); sol.fields(); t_fields = _pc() - t0
    tp, tg = [], []
    for a, b in ring_a:
        t0 = _pc(); sol.price(a, b); tp.append(_pc() - t0)
    for a, b in ring_b:
        t0 = _pc()
        (sol.greeks(a, b) if hasattr(sol, "greeks") else sol.grid_greeks(a, b))
        tg.append(_pc() - t0)
    g = sol.greeks if hasattr(sol, "greeks") else sol.grid_greeks
    skipped = 0
    t0 = _pc()
    for a, b in zip(bS1, bS2):
        try:
            g(a, b)
        except ValueError:              # within about a cell of the barrier
            skipped += 1
    t_batch = _pc() - t0
    return dict(fields_once=t_fields, fresh_price=_stats(tp), fresh_price_greeks=_stats(tg),
                batch_points=int(len(bS1)), batch_skipped_at_barrier=skipped,
                batch_greeks_total=t_batch,
                batch_greeks_per_point=t_batch / max(len(bS1) - skipped, 1))


def fd_costs(name, spec_solver_for_spots):
    market, payoff, domain = _contract(name)
    ring_a = _ring(spec_solver_for_spots, 0.04, N_FRESH, 0.1)
    ring_b = _ring(spec_solver_for_spots, 0.07, N_FRESH, 0.37)
    bS1, bS2 = _batch_spots(spec_solver_for_spots)
    out = dict(levels=[])
    curved = name in CURVED
    levels = Z_LEVELS if curved else X_LEVELS
    for lev, Nt in levels:
        reps = REPEATS if lev <= 160 else (2 if curved else 1)
        walls, sol = [], None
        for k in range(reps + 1):
            sol = (fdm_z_solve(market, domain, payoff, m=lev, N_time=Nt) if curved
                   else fdm_x_onesolve(market, domain, payoff, N=lev, N_time=Nt,
                                       permc_spec=X_ORDERING))
            if k:
                walls.append(sol.wall)
        row = dict(level=lev, N_time=Nt,
                   n_unknowns=int(sol.grid.n_unknowns if curved else sol.inside.sum()),
                   wall={k: _stats([w[k] for w in walls]) for k in walls[0]
                         if k != "n_unknowns"})
        if lev == (160 if curved else 320):
            row["evaluation"] = _fd_eval_costs(sol, market, ring_a, ring_b, bS1, bS2)

            def market_greeks(lev=lev, Nt=Nt):
                for kw in (dict(sigma1=market.sigma1 + 0.01), dict(sigma1=market.sigma1 - 0.01),
                           dict(sigma2=market.sigma2 + 0.01), dict(sigma2=market.sigma2 - 0.01),
                           dict(r=market.r + 1e-4), dict(r=market.r - 1e-4)):
                    mk = _bumped(market, **kw)
                    if curved:
                        fdm_z_solve(mk, domain, payoff, m=lev, N_time=Nt).price()
                    else:
                        fdm_x_onesolve(mk, domain, payoff, N=lev, N_time=Nt,
                                       permc_spec=X_ORDERING).price()
            row["market_greeks"] = _repeat(market_greeks, 1)
            if not curved:
                # the production solver's own ordering, for continuity with Table 6.3
                row["production_ordering"] = _stats(
                    [fdm_x_onesolve(market, domain, payoff, N=lev, N_time=Nt).wall["total_s"]
                     for _ in range(2)])
        out["levels"].append(row)
        print(f"   FD level {lev}: {row['wall']['total_s']['median']:.3f}s "
              f"({row['n_unknowns']} unknowns)", flush=True)
        del sol
    if curved:
        out["levels_fast"] = []
        for lev, Nt in Z_LEVELS_FAST:
            reps = REPEATS if lev <= 160 else 2
            walls, sol = [], None
            for k in range(reps + 1):
                sol = fdm_z_solve(market, domain, payoff, m=lev, N_time=Nt)
                if k:
                    walls.append(sol.wall)
            g = sol.greeks()
            out["levels_fast"].append(dict(
                level=lev, N_time=Nt, n_unknowns=int(sol.grid.n_unknowns),
                wall={k: _stats([w[k] for w in walls]) for k in walls[0] if k != "n_unknowns"},
                values={k: float(g[k]) for k in KEYS}))
            print(f"   FD fast level {lev}/{Nt}: {out['levels_fast'][-1]['wall']['total_s']['median']:.3f}s",
                  flush=True)
            del sol
    return out


def run(names):
    mathieu_basis._ROOT_CACHE = _NoRootMemo()
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data["hardware"] = _hardware()
    data["protocol"] = dict(repeats=REPEATS, warmup=1, fresh_spots=N_FRESH,
                            root_memo="off", threads=1)
    rows = data.setdefault("rows", {})
    for name in names:
        print(f"[cost2] {name}", flush=True)
        t0 = _pc()
        spec = spectral_costs(name)
        print(f"   spectral cold {spec['cold']['median']:.3f}s market "
              f"{spec['market_greeks']['median']:.2f}s", flush=True)
        _c, s = build(name)
        s.fit(); s.project()
        fd = fd_costs(name, s)
        rows[name] = dict(spectral=spec, fd=fd, fd_method="z-space" if name in CURVED
                          else "x-space", wall_s=_pc() - t0)
        data["provenance"] = build_provenance(section="cost2")
        OUT.write_text(json.dumps(data, indent=2, default=float))
        print(f"   done in {_pc() - t0:.0f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  matched: one FD solve on the coarsest grid that matches the spectral price
# ═══════════════════════════════════════════════════════════════════════
#
# The cost comparison times ONE finite-difference solve,
# on the coarsest grid whose price error against the three-grid reference is at most
# that of the spectral price (no Richardson step in the timed FD). The grid is found by
# scanning down from the finest reference grid in fixed steps and stopping at the first
# grid that no longer matches, so every finer grid of the scan matches as well and a
# chance cancellation on one grid cannot set the level. Time steps stay proportional to
# the spacing, as in Z_LEVELS and X_LEVELS.

MATCHED = ART / "paper1_cost2_matched.json"
Z_SCAN = tuple(range(320, 90, -10))         # m, steps from the centre to the box edge
X_SCAN = tuple(range(320, 30, -20))         # N, grid points per axis


def _nt(curved, level):
    return int(np.ceil(25 * level / (8 if curved else 16)))


def _references():
    """Reference values (curved: z-space, rectangles: x-space, both three-grid
    Richardson) and the relative error of the production spectral values."""
    lv = json.loads((ART / "paper1_zfdm_levels.json").read_text())
    xg = json.loads((ART / "paper1_zfdm_xgreeks.json").read_text())["rectangles"]
    spec_g = {r["config"]: r["greeks"] for r in
              json.loads((ART / "paper1_evidence.json").read_text())["cost"]}
    out = {}
    for name in ALL_CONFIGS:
        ex = lv[f"{name}/cholesky"]["extrapolated"] if name in CURVED else xg[name]["extrapolated"]
        ref = {k: ex[k]["V_R"] for k in KEYS}
        out[name] = (ref, {k: (spec_g[name][k] - ref[k]) / ref[k] for k in KEYS},
                     {k: ex[k]["e"] / abs(ref[k]) for k in KEYS})
    return out


def _fd_spot_values(name, market, payoff, domain, level, walls=None):
    """Price and spot Greeks at the spot from one solve (the read-out of the references).
    If `walls` is a list, the solve's wall-time split is appended to it."""
    if name in CURVED:
        sol = fdm_z_solve(market, domain, payoff, m=level, N_time=_nt(True, level))
        g = sol.greeks()
    else:
        sol = fdm_x_onesolve(market, domain, payoff, N=level, N_time=_nt(False, level),
                             permc_spec=X_ORDERING)
        g = sol.grid_greeks()
    if walls is not None:
        walls.append(dict(sol.wall))
    return {k: float(g[k]) for k in KEYS}


def _fd_market_greeks(name, market, payoff, domain, level):
    """The six re-solved markets of the vegas and rho (bumps 0.01 and 1e-4)."""
    for kw in (dict(sigma1=market.sigma1 + 0.01), dict(sigma1=market.sigma1 - 0.01),
               dict(sigma2=market.sigma2 + 0.01), dict(sigma2=market.sigma2 - 0.01),
               dict(r=market.r + 1e-4), dict(r=market.r - 1e-4)):
        _fd_spot_values(name, _bumped(market, **kw), payoff, domain, level)


def matched(names):
    refs = _references()
    data = json.loads(MATCHED.read_text()) if MATCHED.exists() else {}
    data["hardware"] = _hardware()
    data["protocol"] = dict(repeats=REPEATS, warmup=1, threads=1, z_scan=Z_SCAN,
                            x_scan=X_SCAN, rule="coarsest grid of a downward scan whose "
                            "price error, and that of every finer scanned grid, is at most "
                            "the spectral price error")
    rows = data.setdefault("rows", {})
    for name in names:
        curved = name in CURVED
        market, payoff, domain = _contract(name)
        ref, spec_err, ref_e = refs[name]
        scan, level = [], None
        for lev in (Z_SCAN if curved else X_SCAN):
            vals = _fd_spot_values(name, market, payoff, domain, lev)
            err = {k: (vals[k] - ref[k]) / ref[k] for k in KEYS}
            ok = abs(err["price"]) <= abs(spec_err["price"])
            scan.append(dict(level=lev, N_time=_nt(curved, lev), rel_err=err, matches=ok))
            print(f"[matched {name}] level {lev}: price error {err['price']:+.2e} "
                  f"(spectral {spec_err['price']:+.2e}) {'ok' if ok else 'stop'}", flush=True)
            if not ok:
                break
            level = lev
        row = dict(fd_method="z-space" if curved else "x-space", spectral_rel_err=spec_err,
                   reference_e_rel=ref_e, scan=scan, matched_level=level,
                   scan_exhausted=bool(scan and scan[-1]["matches"]))
        if level is not None:
            row["matched_N_time"] = _nt(curved, level)
            row["matched_rel_err"] = next(s["rel_err"] for s in scan if s["level"] == level)
            walls = []
            row["price_and_spot_greeks_s"] = _repeat(
                lambda: _fd_spot_values(name, market, payoff, domain, level, walls))
            walls = walls[1:]                                # drop the warm-up
            # a new payoff on the same grid and market reuses the assembly and the LU
            row["wall_split"] = {k: _stats([w[k] for w in walls]) for k in walls[0]
                                 if k != "n_unknowns"}
            row["market_greeks_s"] = _repeat(
                lambda: _fd_market_greeks(name, market, payoff, domain, level), 2)
            print(f"   matched level {level}: "
                  f"{row['price_and_spot_greeks_s']['median']:.3f}s", flush=True)
        rows[name] = row
        data["provenance"] = build_provenance(section="cost2m")
        MATCHED.write_text(json.dumps(data, indent=2, default=float))


# ═══════════════════════════════════════════════════════════════════════
#  report: matched accuracy
# ═══════════════════════════════════════════════════════════════════════

def _matched_rows():
    """The matched single-grid level, its errors and its cost, if `matched` has run."""
    if not MATCHED.exists():
        return {}
    rows = json.loads(MATCHED.read_text())["rows"]
    return {n: dict(level=r["matched_level"], N_time=r.get("matched_N_time"),
                    scan_exhausted=r["scan_exhausted"], rel_err=r.get("matched_rel_err"),
                    price_and_spot_greeks_s=(r["price_and_spot_greeks_s"]["median"]
                                             if "price_and_spot_greeks_s" in r else None),
                    new_payoff_lu_reused_s=(
                        r["wall_split"]["steps_s"]["median"]
                        + r["wall_split"].get("payoff_s", {}).get("median", 0.0)
                        if "wall_split" in r else None),
                    market_greeks_s=(r["market_greeks_s"]["median"]
                                     if "market_greeks_s" in r else None))
            for n, r in rows.items()}


def report():
    cost = json.loads(OUT.read_text())["rows"]
    lv = json.loads((ART / "paper1_zfdm_levels.json").read_text())
    xg = json.loads((ART / "paper1_zfdm_xgreeks.json").read_text())["rectangles"]
    # Production spectral values after the 26 Oct 2026 mode change (paper1_greeks.json
    # predates it for dnt and worst_of).
    spec_g = {r["config"]: r["greeks"] for r in
              json.loads((ART / "paper1_evidence.json").read_text())["cost"]}
    keys = ("price", "delta1", "delta2", "gamma1", "gamma2", "gamma12", "theta")
    out = {}
    for name in ALL_CONFIGS:
        if name not in cost:
            continue
        c = cost[name]
        ex = (lv[f"{name}/cholesky"]["extrapolated"] if name in CURVED
              else xg[name]["extrapolated"])
        fd_vals = (lv[f"{name}/cholesky"]["levels"] if name in CURVED else xg[name]["levels"])
        ref = {k: ex[k]["V_R"] for k in keys}
        spec_err = {k: (spec_g[name][k] - ref[k]) / ref[k] for k in keys}
        levels = []
        for row, vals in zip(c["fd"]["levels"], fd_vals):
            err = {k: (vals[k] - ref[k]) / ref[k] for k in keys}
            w = row["wall"]
            levels.append(dict(level=row["level"], n_unknowns=row["n_unknowns"],
                               one_solve_s=w["total_s"]["median"],
                               # a new payoff on the same grid and market reuses the LU
                               new_payoff_lu_reused_s=(w["steps_s"]["median"]
                                                       + w.get("payoff_s", {}).get("median", 0.0)),
                               rel_err=err))
        fast = []
        for row in c["fd"].get("levels_fast", []):
            fast.append(dict(level=row["level"], N_time=row["N_time"],
                             one_solve_s=row["wall"]["total_s"]["median"],
                             rel_err={k: (row["values"][k] - ref[k]) / ref[k] for k in keys}))

        def two_grid(rows_cost, rows_vals):
            """Richardson with p = 2 from consecutive levels; cost of both solves."""
            out2 = []
            for a, b, va, vb in zip(rows_cost, rows_cost[1:], rows_vals, rows_vals[1:]):
                rv = {k: vb[k] + (vb[k] - va[k]) / 3.0 for k in keys}
                out2.append(dict(levels=(a["level"], b["level"]),
                                 cost_s=a["one_solve_s"] + b["one_solve_s"],
                                 rel_err={k: (rv[k] - ref[k]) / ref[k] for k in keys}))
            return out2[:-1] if name in CURVED else out2   # the last pair built V_R itself
        two = two_grid(levels, fd_vals)
        if fast:
            two_fast = two_grid(fast, [r["values"] for r in c["fd"]["levels_fast"]])
        else:
            two_fast = []
        match = next((L for L in levels if abs(L["rel_err"]["price"])
                      <= abs(spec_err["price"])), None)
        sp = c["spectral"]
        refined = None
        if "refined" in sp:
            refined = dict(cold_s=sp["refined"]["cold"]["median"],
                           rel_err={k: (sp["refined"]["values"][k] - ref[k]) / ref[k]
                                    for k in keys})
        out[name] = dict(
            reference_e_rel={k: ex[k]["e"] / abs(ref[k]) for k in keys},
            spectral=dict(cold_s=sp["cold"]["median"], fit_s=sp["fit"]["median"],
                          project_s=sp["project"]["median"],
                          fresh_price_s=sp["fresh_price"]["median"],
                          fresh_price_greeks_s=sp["fresh_price_greeks"]["median"],
                          batch_per_point_s=sp["batch"]["per_point_median"],
                          market_greeks_s=sp["market_greeks"]["median"],
                          # 28 Oct only: rho by rebuilding the basis, a code path the
                          # current solver no longer has (rho always reuses the basis).
                          market_greeks_rate_rebuild_s=(sp["market_greeks_rate_rebuild"]["median"]
                                                        if "market_greeks_rate_rebuild" in sp
                                                        else None),
                          roots_cached=(dict(
                              solve_s=sp["roots_cached"]["cold_with_cached_roots"]["median"],
                              fit_s=sp["roots_cached"]["fit"]["median"],
                              all_hits=sp["roots_cached"]["all_hits"],
                              memo_source=sp["roots_cached"]["memo_source"],
                              same_contract_solve_s=(
                                  sp["roots_cached"]["same_contract"]
                                  ["cold_with_cached_roots"]["median"]
                                  if "same_contract" in sp["roots_cached"] else None))
                              if "roots_cached" in sp else None),
                          new_payoff_on_basis_s=sp["project"]["median"],
                          rel_err=spec_err),
            spectral_refined_quadrature=refined,
            fd_levels=levels,
            fd_levels_fast=fast,
            fd_two_grid=two,
            fd_two_grid_fast=two_fast,
            fd_first_level_matching_spectral_price_error=match,
            fd_evaluation=next(r["evaluation"] for r in c["fd"]["levels"] if "evaluation" in r),
            fd_market_greeks_s=next(r["market_greeks"]["median"] for r in c["fd"]["levels"]
                                    if "market_greeks" in r),
            fd_matched_single_grid=(_matched_rows().get(name)),
        )
        print(name, json.dumps(out[name]["spectral"]["rel_err"]), flush=True)
    REPORT.write_text(json.dumps(dict(rows=out, provenance=build_provenance(section="cost2r")),
                                 indent=2, default=float))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    cmd, *names = sys.argv[1:]
    if cmd == "run":
        run(names or list(ALL_CONFIGS))
    elif cmd == "matched":
        matched(names or list(ALL_CONFIGS))
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
