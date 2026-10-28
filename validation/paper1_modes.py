"""Mode-count convergence of the double no-touch and the worst-of call."""
from __future__ import annotations

import json
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

from paper1_evidence import build
from twoasset.runlog import build_provenance
from twoasset.solvers._optimal import ARTIFACTS_DIR, OPTIMAL

ART = Path(ARTIFACTS_DIR)
OUT = ART / "paper1_modes.json"
CANDIDATES = {"dnt": (10, 12, 14, 16, 20), "worst_of": (20, 24, 28, 32)}
SPOT = ("delta1", "delta2", "gamma1", "gamma2", "gamma12", "theta")
MARKET = ("vega1", "vega2", "rho")


def main():
    table = {r["config"]: r for r in json.loads((ART / "paper1_price_table.json").read_text())["rows"]}
    rows = []
    for name, Ns in CANDIDATES.items():
        fd = table[name]["fd"]
        mc = table[name]["mc"]
        fdg = tomllib.loads((ROOT / "validation" / f"{name}.toml").read_text())["fdm_greeks"]
        for N in Ns:
            _cfg, solver = build(name, n_modes=N)
            t0 = time.perf_counter(); solver.fit(); solver.project()
            cold = time.perf_counter() - t0
            price = float(solver.price())
            t0 = time.perf_counter(); g = solver.greeks(); t_spot = time.perf_counter() - t0
            t0 = time.perf_counter(); gf = solver.greeks(full=True); t_full = time.perf_counter() - t0
            greeks = {k: float(gf[k]) for k in SPOT + MARKET}
            rel = {k: abs(greeks[k] - fdg[k]) / abs(fdg[k]) for k in SPOT + MARKET}
            row = dict(
                config=name, N=N, n_modes=int(solver.n_modes()),
                production=(N == OPTIMAL[name]["N_modes"]),
                cold_s=cold, spot_greeks_s=t_spot, full_greeks_s=t_full,
                one_price_s=cold, all_nine_s=cold + t_spot + t_full,
                price=price,
                vs_fd=dict(pct=100 * (price - fd["V_R"]) / fd["V_R"],
                           u=(price - fd["V_R"]) / fd["u"]),
                vs_mc=dict(pct=100 * (price - mc["mean"]) / mc["mean"],
                           sem=(price - mc["mean"]) / mc["sem"]),
                greeks=greeks, greeks_rel_vs_fd=rel,
                max_spot_rel=max(rel[k] for k in SPOT),
                max_market_rel=max(rel[k] for k in MARKET),
            )
            rows.append(row)
            print(f"{name:9s} {row['n_modes']:5d} modes  cold {cold:6.3f}s  all-nine {row['all_nine_s']:6.2f}s  "
                  f"price {price:.7f}  vs FD {row['vs_fd']['pct']:+.4f}% ({row['vs_fd']['u']:+.1f} u)  "
                  f"vs MC {row['vs_mc']['sem']:+.2f} SEM  max spot-Greek diff {100*row['max_spot_rel']:.2f}%  "
                  f"max market-Greek diff {100*row['max_market_rel']:.2f}%", flush=True)
    OUT.write_text(json.dumps(dict(rows=rows, provenance=build_provenance()), indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
