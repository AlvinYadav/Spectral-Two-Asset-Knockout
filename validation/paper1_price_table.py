"""Table 6.1: spectral prices against Monte Carlo and three-grid finite differences."""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twoasset.cli import assemble
from twoasset.runlog import build_provenance
from twoasset.solvers._optimal import ALL_CONFIGS, ARTIFACTS_DIR
from twoasset.solvers.fdm import fdm_price

ART = Path(ARTIFACTS_DIR)
OUT = ART / "paper1_price_table.json"


def band(z):
    z = abs(z)
    return "green" if z <= 2 else ("orange" if z <= 3 else "red")


def main():
    spec = {r["config"]: r["price"] for r in
            json.loads((ART / "paper1_evidence.json").read_text())["cost"]}
    mc = {r["config"]: r for r in json.loads((ART / "paper1_mc.json").read_text())["rows"]}
    shift = json.loads((ART / "paper1_fdm_shift.json").read_text())["rows"]
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    v160_cache = {r["config"]: r["fd"]["V160"] for r in old.get("rows", [])}
    rows = []
    for name in ALL_CONFIGS:
        stored = tomllib.loads((ROOT / "validation" / f"{name}.toml").read_text())
        if name in v160_cache:
            v160 = v160_cache[name]
        else:
            cfg = tomllib.loads((ROOT / stored["config"]).read_text())
            market, payoff, domain, _b, _ = assemble(cfg)
            v160 = float(fdm_price(market, domain, payoff=payoff, N=160, N_time=250)["price"])
        v320, v640 = stored["fdm"]["coarse_price"], stored["fdm"]["fine_price"]
        p = float(np.log2((v160 - v320) / (v320 - v640)))
        vr = v640 + (v640 - v320) / (2.0 ** p - 1.0)
        u_fd = abs(vr - v640)
        s = spec[name]
        if name not in mc:
            print(f"{name}: no Monte Carlo runs merged yet")
            continue
        m = mc[name]
        d_mc, d_fd = s - m["mean"], s - vr
        row = dict(
            config=name, spectral=s,
            mc=dict(mean=m["mean"], sem=m["sem"], n_runs=m["n_runs"],
                    diff_pct=100 * d_mc / m["mean"], diff_u=d_mc / m["sem"],
                    band=band(d_mc / m["sem"])),
            fd=dict(V160=v160, V320=v320, V640=v640, p=p, V_R=vr, u=u_fd,
                    u_pct=100 * u_fd / vr, diff_pct=100 * d_fd / vr, diff_u=d_fd / u_fd,
                    band=band(d_fd / u_fd)),
        )
        key = f"{name}/N320"
        if key in shift and shift[key].get("sd"):
            sh = shift[key]
            row["placement"] = dict(n=sh["n"], mean=sh["mean"], sd=sh["sd"], sem=sh["sem"],
                                    sd_pct=100 * sh["sd"] / sh["mean"],
                                    refinement_error=abs(v320 - vr),
                                    ratio=abs(v320 - vr) / sh["sd"])
        rows.append(row)
        print(f"{name:14s} spec {s:.6f} | MC {m['mean']:.6f}+-{m['sem']:.6f} "
              f"({row['mc']['diff_pct']:+.3f}%, {row['mc']['diff_u']:+.2f} SEM) | "
              f"FD V_R {vr:.6f} u {u_fd:.2e} p {p:.2f} "
              f"({row['fd']['diff_pct']:+.4f}%, {row['fd']['diff_u']:+.2f} u)"
              + (f" | placement sd {row['placement']['sd']:.1e} "
                 f"(refinement error / sd = {row['placement']['ratio']:.0f})"
                 if "placement" in row else ""))
    OUT.write_text(json.dumps(dict(rows=rows, provenance=build_provenance()), indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
