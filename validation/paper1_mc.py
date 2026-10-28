"""Monte Carlo reference prices: twenty runs of 2e6 paths per contract, pooled."""
from __future__ import annotations

import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twoasset.cli import assemble
from twoasset.runlog import build_provenance
from twoasset.solvers._optimal import ALL_CONFIGS, ARTIFACTS_DIR
from twoasset.solvers.mc import mc_price

N_PATHS = 2_000_000
N_STEPS = 500
CHECK_SEED = 1
NEW_SEEDS = list(range(1001, 1016))      # fifteen new runs; twenty in total

PARTS = Path(ARTIFACTS_DIR) / "paper1_mc_parts"
OUT = Path(ARTIFACTS_DIR) / "paper1_mc.json"


def _stored(name):
    return tomllib.loads((ROOT / "validation" / f"{name}.toml").read_text())


def _one(market, payoff, domain, seed):
    t0 = time.perf_counter()
    r = mc_price(market, payoff, domain, n_paths=N_PATHS, n_steps=N_STEPS,
                 seed=seed, use_brownian_bridge=True, antithetic=True)
    return dict(seed=seed, price=float(r["price"]), std_err=float(r["std_err"]),
                wall_s=time.perf_counter() - t0)


def run(name):
    stored = _stored(name)
    cfg = tomllib.loads((ROOT / stored["config"]).read_text())
    market, payoff, domain, _basis, _ = assemble(cfg)
    PARTS.mkdir(parents=True, exist_ok=True)
    part = PARTS / f"{name}.json"
    data = json.loads(part.read_text()) if part.exists() else {}
    data.update(config=name, n_paths=N_PATHS, n_steps=N_STEPS)

    if "check" not in data:
        c = _one(market, payoff, domain, CHECK_SEED)
        i = stored["mc"]["seeds"].index(CHECK_SEED)
        c["stored_price"] = float(stored["mc"]["prices"][i])
        c["abs_diff"] = abs(c["price"] - c["stored_price"])
        data["check"] = c
        part.write_text(json.dumps(data, indent=2))
        print(f"[{name}] check seed {CHECK_SEED}: {c['price']:.10f} vs stored "
              f"{c['stored_price']:.10f} (diff {c['abs_diff']:.2e})", flush=True)

    runs = {r["seed"]: r for r in data.get("runs", [])}
    for s in NEW_SEEDS:
        if s in runs:
            continue
        runs[s] = _one(market, payoff, domain, s)
        data["runs"] = [runs[k] for k in sorted(runs)]
        part.write_text(json.dumps(data, indent=2))
        print(f"[{name}] seed {s}: {runs[s]['price']:.8f} "
              f"({runs[s]['wall_s']:.0f}s)", flush=True)


def merge():
    rows = []
    for name in ALL_CONFIGS:
        part = PARTS / f"{name}.json"
        if not part.exists():
            print(f"[merge] {name}: no runs yet")
            continue
        d = json.loads(part.read_text())
        stored = _stored(name)["mc"]
        new = [r["price"] for r in d.get("runs", [])]
        check_ok = d["check"]["abs_diff"] <= 1e-9 * max(1.0, abs(d["check"]["stored_price"]))
        pooled = (list(stored["prices"]) if check_ok else []) + new
        seeds = (list(stored["seeds"]) if check_ok else []) + [r["seed"] for r in d["runs"]]
        p = np.asarray(pooled)
        mean = float(p.mean())
        sem = float(p.std(ddof=1) / np.sqrt(len(p)))
        rows.append(dict(
            config=name, n_runs=len(p), n_paths_per_run=N_PATHS, n_steps=N_STEPS,
            seeds=seeds, prices=pooled, mean=mean, sem=sem,
            stored_mean=float(stored["mean"]), stored_sem=float(stored["sem"]),
            stored_n_runs=len(stored["prices"]), check=d["check"], check_ok=bool(check_ok),
            mean_wall_s_per_run=float(np.mean([r["wall_s"] for r in d["runs"]])),
        ))
        print(f"[merge] {name}: {len(p)} runs, mean {mean:.6f} +- {sem:.6f} "
              f"(was {stored['mean']:.6f} +- {stored['sem']:.6f}, "
              f"check {'ok' if check_ok else 'FAILED'})")
    OUT.write_text(json.dumps(dict(rows=rows, provenance=build_provenance()), indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    cmd, *names = sys.argv[1:]
    if cmd == "run":
        for n in names:
            run(n)
    elif cmd == "merge":
        merge()
    else:
        raise SystemExit(__doc__)
