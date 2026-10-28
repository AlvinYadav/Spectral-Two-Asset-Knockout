"""Finite-difference prices on randomly translated grids, to measure the effect of
grid placement."""
from __future__ import annotations

import argparse
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
from twoasset.solvers._optimal import ARTIFACTS_DIR
from twoasset.solvers.fdm import fdm_price

CURVED = ("default", "tight_barrier", "mahalanobis")
N_TIME = {80: 125, 160: 250, 320: 500, 640: 1000}       # the paper's grid/step pairs
SEED = 20261005
OUT = Path(ARTIFACTS_DIR) / "paper1_fdm_shift.json"


def offsets(k):
    return np.random.default_rng(SEED).uniform(-0.5, 0.5, size=(k, 2))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int, nargs="*", default=[160, 320])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--configs", nargs="*", default=list(CURVED))
    a = ap.parse_args(argv)
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data.setdefault("rows", {})
    data["seed"], data["note"] = SEED, __doc__.split("\n\n")[1]
    offs = offsets(a.k)
    for name in a.configs:
        cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text())
        market, payoff, domain, _basis, _ = assemble(cfg)
        for N in a.N:
            key = f"{name}/N{N}"
            row = data["rows"].get(key, {"config": name, "N": N, "N_time": N_TIME[N],
                                         "placements": []})
            if "aligned" not in row:
                t0 = time.perf_counter()
                row["aligned"] = float(fdm_price(market, domain, payoff=payoff, N=N,
                                                 N_time=N_TIME[N])["price"])
                row["aligned_wall_s"] = time.perf_counter() - t0
            done = len(row["placements"])
            for o in offs[done:]:
                t0 = time.perf_counter()
                p = fdm_price(market, domain, payoff=payoff, N=N, N_time=N_TIME[N],
                              offset=(float(o[0]), float(o[1])))["price"]
                row["placements"].append(dict(offset=[float(o[0]), float(o[1])],
                                              price=float(p),
                                              wall_s=time.perf_counter() - t0))
                prices = np.array([q["price"] for q in row["placements"]])
                row.update(n=len(prices), mean=float(prices.mean()),
                           sd=float(prices.std(ddof=1)) if len(prices) > 1 else None,
                           sem=float(prices.std(ddof=1) / np.sqrt(len(prices)))
                           if len(prices) > 1 else None)
                data["rows"][key] = row
                data["provenance"] = build_provenance()
                OUT.write_text(json.dumps(data, indent=2))
                print(f"[{key}] {row['n']:2d}: {p:.6f}  mean {row['mean']:.6f}"
                      + (f"  sem {row['sem']:.2e}" if row["sem"] else ""), flush=True)
            print(f"[{key}] done: aligned {row['aligned']:.6f}, mean {row['mean']:.6f}, "
                  f"sd {row['sd']:.2e}, sem {row['sem']:.2e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
