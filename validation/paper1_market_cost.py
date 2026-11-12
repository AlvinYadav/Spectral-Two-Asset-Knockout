"""Cost of the three market Greeks after the rate bumps were moved onto the fitted
eigenbasis: the price with all nine Greeks at the valuation spot on a solved basis,
as in the greeks_full timing of paper1_evidence.py, with the Mathieu root memo off
so that every volatility bump pays its own root scan."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

import twoasset.bases.mathieu as mathieu_basis
from paper1_evidence import ALL_CONFIGS, _NoRootMemo, _pc, build
from twoasset.runlog import build_provenance
from twoasset.solvers._optimal import ARTIFACTS_DIR

OUT = Path(ARTIFACTS_DIR) / "paper1_market_cost.json"


def market_cost_one(name):
    _, solver = build(name)
    solver.fit()
    solver.project()
    solver.greeks()                       # the spot Greeks are then a hot valuation
    t0 = _pc(); g = solver.greeks(full=True); t = _pc() - t0
    print(f"[market] {name:14s} {t:.3f}s", flush=True)
    return dict(config=name, greeks_full_s=t,
                greeks_full={k: float(v) for k, v in g.items()})


def main():
    mathieu_basis._ROOT_CACHE = _NoRootMemo()
    rows = [market_cost_one(n) for n in ALL_CONFIGS]
    OUT.write_text(json.dumps(dict(
        rows=rows, provenance=build_provenance(),
        notes=["greeks_full_s: four re-solved volatility markets and two rate bumps "
               "projected on the fitted eigenbasis, plus the hot spot Greeks."]),
        indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
