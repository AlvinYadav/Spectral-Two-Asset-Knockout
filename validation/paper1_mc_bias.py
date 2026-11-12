"""Monitoring bias of the Monte Carlo benchmark (Paper 1).

The production benchmark (`mc_price`, 500 steps) applies the tangent-plane Brownian-
bridge correction between steps. Its standard error measures sampling noise only. This
script estimates the remaining time-step (monitoring) bias with coupled paths, with a
design fixed before the run.

Each path is simulated with N_FINE = 2000 exact log-normal steps and antithetic pairs.
The coarser levels n = 1000, 500 and 250 use the SAME Brownian path, observed every 2, 4
and 8 fine steps, and the bridge factor of `mc.py` is applied between consecutive
observations at each level. The increments are exact, so the levels differ only in the
monitoring correction, and the level differences have small variance.

For `default`, the common-random-number deltas (+-1% of spot, start offsets as in
`mc_delta_crn`) are carried along at every level.

Standard errors are computed from antithetic-pair averages, which are independent.

Run (repo root):
    PYTHONPATH=. .venv/bin/python3 validation/paper1_mc_bias.py run CONFIG [CONFIG ...]
    PYTHONPATH=. .venv/bin/python3 validation/paper1_mc_bias.py merge
Writes:
    validation/artifacts/paper1_mc_bias_parts/{config}.json   (per batch, as it runs)
    validation/artifacts/paper1_mc_bias.json                  (merge)
"""
from __future__ import annotations

import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twoasset.cli import assemble                                   # noqa: E402
from twoasset.core.domain import EuclideanCircle, MahalanobisEllipse, Rectangle  # noqa: E402
from twoasset.runlog import build_provenance                        # noqa: E402
from twoasset.solvers._optimal import ALL_CONFIGS, ARTIFACTS_DIR    # noqa: E402
from twoasset.solvers.mc import (_bridge_step_euclidean_circle,     # noqa: E402
                                 _bridge_step_mahalanobis, _bridge_step_rectangle)

N_FINE = 2000
BLOCKS = (1, 2, 4, 8)                  # levels n = 2000, 1000, 500, 250
N_PATHS = 500_000                      # per batch
SEEDS = (2001, 2002, 2003, 2004)       # four batches per contract
DELTA_CONFIGS = ("default",)
BUMP_PCT = 1.0

PARTS = Path(ARTIFACTS_DIR) / "paper1_mc_bias_parts"
OUT = Path(ARTIFACTS_DIR) / "paper1_mc_bias.json"


def _bridge(market, domain):
    if isinstance(domain, EuclideanCircle):
        return lambda a1, a2, b1, b2, dt: _bridge_step_euclidean_circle(
            a1, a2, b1, b2, domain.R, market.Sigma, dt)
    if isinstance(domain, MahalanobisEllipse):
        return lambda a1, a2, b1, b2, dt: _bridge_step_mahalanobis(
            a1, a2, b1, b2, domain.R, market, dt)
    if isinstance(domain, Rectangle):
        return lambda a1, a2, b1, b2, dt: _bridge_step_rectangle(
            a1, a2, b1, b2, domain, market.Sigma, dt)
    raise ValueError(type(domain).__name__)


def _offsets(market, with_deltas):
    """Start offsets (x1, x2): base, then +-h on asset 1 and on asset 2."""
    if not with_deltas:
        return [(0.0, 0.0)]
    f = BUMP_PCT / 100.0
    up, dn = np.log1p(f), np.log1p(-f)
    return [(0.0, 0.0), (up, 0.0), (dn, 0.0), (0.0, up), (0.0, dn)]


def _pair_stats(s, n_half):
    a = 0.5 * (s[:n_half] + s[n_half:])          # antithetic-pair averages, i.i.d.
    return float(a.mean()), float(a.var(ddof=1))


def run_batch(market, payoff, domain, seed, with_deltas):
    bridge = _bridge(market, domain)
    offsets = _offsets(market, with_deltas)
    rng = np.random.default_rng(seed)
    n_half = N_PATHS // 2
    n = 2 * n_half
    dt = market.T / N_FINE
    s1, s2, rho = market.sigma1, market.sigma2, market.rho
    lchol = np.array([[s1, 0.0], [rho * s2, np.sqrt(1 - rho * rho) * s2]])
    drift = np.array([market.r - 0.5 * s1 * s1, market.r - 0.5 * s2 * s2])

    x1 = np.zeros(n)
    x2 = np.zeros(n)
    prev = {B: (x1, x2) for B in BLOCKS}
    surv = {(B, j): np.ones(n) for B in BLOCKS for j in range(len(offsets))}
    for k in range(N_FINE):
        zh = rng.standard_normal((2, n_half))
        dw = lchol @ np.concatenate([zh, -zh], axis=1) * np.sqrt(dt)
        x1 = x1 + drift[0] * dt + dw[0]
        x2 = x2 + drift[1] * dt + dw[1]
        for B in BLOCKS:
            if (k + 1) % B:
                continue
            p1, p2 = prev[B]
            for j, (o1, o2) in enumerate(offsets):
                surv[(B, j)] *= bridge(p1 + o1, p2 + o2, x1 + o1, x2 + o2, B * dt)
            prev[B] = (x1, x2)

    disc = np.exp(-market.r * market.T)
    raw = [np.asarray(payoff(market.S1_0 * np.exp(x1 + o1), market.S2_0 * np.exp(x2 + o2)),
                      dtype=float) for o1, o2 in offsets]
    h1 = market.S1_0 * BUMP_PCT / 100.0
    h2 = market.S2_0 * BUMP_PCT / 100.0

    def samples(B):
        out = {"price": disc * surv[(B, 0)] * raw[0]}
        if with_deltas:
            out["delta1"] = disc * (surv[(B, 1)] * raw[1] - surv[(B, 2)] * raw[2]) / (2 * h1)
            out["delta2"] = disc * (surv[(B, 3)] * raw[3] - surv[(B, 4)] * raw[4]) / (2 * h2)
        return out

    fine = samples(1)
    levels = {}
    for B in BLOCKS:
        cur = samples(B)
        lv = {}
        for q, s in cur.items():
            m, v = _pair_stats(s, n_half)
            dm, dv = _pair_stats(s - fine[q], n_half)
            lv[q] = dict(mean=m, var=v, diff_mean=dm, diff_var=dv)
        levels[str(N_FINE // B)] = lv
    return dict(seed=seed, n_pairs=n_half, levels=levels)


def run(name):
    cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text())
    market, payoff, domain, _b, _ = assemble(cfg)
    PARTS.mkdir(parents=True, exist_ok=True)
    part = PARTS / f"{name}.json"
    data = json.loads(part.read_text()) if part.exists() else {}
    data.update(config=name, n_fine=N_FINE, blocks=list(BLOCKS), n_paths_per_batch=N_PATHS,
                bump_pct=BUMP_PCT, deltas=name in DELTA_CONFIGS)
    batches = {b["seed"]: b for b in data.get("batches", [])}
    for seed in SEEDS:
        if seed in batches:
            continue
        t0 = time.perf_counter()
        b = run_batch(market, payoff, domain, seed, name in DELTA_CONFIGS)
        b["wall_s"] = time.perf_counter() - t0
        batches[seed] = b
        data["batches"] = [batches[s] for s in sorted(batches)]
        part.write_text(json.dumps(data, indent=2))
        lv = b["levels"]
        print(f"[{name}] seed {seed} ({b['wall_s']:.0f}s): "
              + "  ".join(f"n={n}: {lv[n]['price']['mean']:.6f}" for n in sorted(lv, key=int)),
              flush=True)


def _pool(batches, n, q):
    """Pool equal-size batches: mean of means, SE from the pooled pair variance."""
    rows = [b["levels"][n][q] for b in batches]
    npairs = sum(b["n_pairs"] for b in batches)
    mean = float(np.mean([r["mean"] for r in rows]))
    dmean = float(np.mean([r["diff_mean"] for r in rows]))
    se = float(np.sqrt(np.mean([r["var"] for r in rows]) / npairs))
    dse = float(np.sqrt(np.mean([r["diff_var"] for r in rows]) / npairs))
    batch_sd = float(np.std([r["diff_mean"] for r in rows], ddof=1)) if len(rows) > 1 else None
    return dict(mean=mean, se=se, diff_to_fine=dmean, diff_se=dse, diff_batch_sd=batch_sd)


def _bias_at_500(pooled):
    """Bias of the 500-step estimator: (V500 - V2000) plus the tail beyond 2000 steps.

    The tail uses the order fitted from d(500,1000) and d(1000,2000) when both differ
    from zero by more than two standard errors; otherwise first order is assumed, which
    gives the larger tail for a decaying bias.
    """
    v = {n: pooled[n]["mean"] for n in pooled}
    d_500_1000 = v["500"] - v["1000"]
    d_1000_2000 = pooled["1000"]["diff_to_fine"]
    se_1000_2000 = pooled["1000"]["diff_se"]
    se_500_1000 = np.hypot(pooled["500"]["diff_se"], se_1000_2000)
    alpha = None
    if (abs(d_500_1000) > 2 * se_500_1000 and abs(d_1000_2000) > 2 * se_1000_2000
            and d_500_1000 * d_1000_2000 > 0):
        alpha = float(np.log2(d_500_1000 / d_1000_2000))
    a = alpha if (alpha is not None and alpha > 0.25) else 1.0
    tail = d_1000_2000 / (2.0 ** a - 1.0)
    bias = pooled["500"]["diff_to_fine"] + tail
    se = float(np.hypot(pooled["500"]["diff_se"], se_1000_2000 / (2.0 ** a - 1.0)))
    return dict(alpha_fitted=alpha, alpha_used=a, tail=float(tail), bias_500=float(bias),
                bias_500_se=se)


def merge():
    sem20 = {r["config"]: r["sem"] for r in
             json.loads((Path(ARTIFACTS_DIR) / "paper1_mc.json").read_text())["rows"]}
    # Table 6.2 Monte Carlo deltas of the Euclidean basket and their standard errors
    crn = json.loads((ROOT / "validation" / "greek_benchmark_default_mc_crn.json").read_text())
    delta_sem = {"delta1": crn["delta1_sem"], "delta2": crn["delta2_sem"]}
    rows = []
    for name in ALL_CONFIGS:
        part = PARTS / f"{name}.json"
        if not part.exists():
            print(f"[merge] {name}: no batches yet")
            continue
        d = json.loads(part.read_text())
        batches = d["batches"]
        qs = list(batches[0]["levels"]["2000"].keys())
        row = dict(config=name, n_batches=len(batches),
                   n_paths=len(batches) * d["n_paths_per_batch"], quantities={})
        for q in qs:
            pooled = {n: _pool(batches, n, q) for n in ("2000", "1000", "500", "250")}
            row["quantities"][q] = dict(levels=pooled, **_bias_at_500(pooled))
        b = row["quantities"]["price"]
        row["sem_20run"] = sem20.get(name)
        if row["sem_20run"]:
            row["bias_over_sem"] = b["bias_500"] / row["sem_20run"]
            row["gate_M1"] = bool(abs(b["bias_500"]) + 2 * b["bias_500_se"]
                                  <= row["sem_20run"] / 3)
        if name == "default":
            for q, sem in delta_sem.items():
                dq = row["quantities"].get(q)
                if dq is not None:
                    dq["sem_table62"] = sem
                    dq["gate_M2"] = bool(abs(dq["bias_500"]) + 2 * dq["bias_500_se"] <= sem / 3)
        rows.append(row)
        print(f"[merge] {name}: bias(500) = {b['bias_500']:+.2e} +- {b['bias_500_se']:.1e} "
              f"(alpha {b['alpha_fitted']}), 20-run SEM {row['sem_20run']:.1e}, "
              f"ratio {row.get('bias_over_sem', float('nan')):+.3f}")
    OUT.write_text(json.dumps(dict(rows=rows, provenance=build_provenance()), indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    cmd, *names = sys.argv[1:]
    if cmd == "run":
        for nm in names:
            run(nm)
    elif cmd == "merge":
        merge()
    else:
        raise SystemExit(__doc__)
