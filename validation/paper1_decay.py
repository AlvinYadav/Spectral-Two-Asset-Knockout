"""Long-maturity decay of the Euclidean basket (Paper 1).

Writes validation/artifacts/paper1_decay.json. Exploratory, no gate.

For rho = 0, 0.5 and 0.8 on the default contract the solved basis is evaluated at the
spot for several maturities, keeping either every mode or only the slowest one. The
basis and the projection do not depend on T (smooth_tau_frac = 0), so one fit serves
every maturity. The long-run decay rate is read off the gauge and the lowest eigenvalue:

    kappa = r + 1/2 alpha' Sigma^-1 alpha + 1/2 lambda_1^2     (V ~ A exp(-kappa T)),

split into discounting, drift and barrier parts.

Run (repo root):  PYTHONPATH=. .venv/bin/python3 validation/paper1_decay.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper1_evidence import build                                     # noqa: E402
from twoasset.solvers._optimal import ARTIFACTS_DIR                  # noqa: E402

RHOS = (0.0, 0.5, 0.8)
MATURITIES = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)


def decay_rows(rho):
    _c, s = build("default", financial={"rho": rho})
    s.fit()
    s.project()
    m, b = s.market, s.basis
    S1, S2 = m.S1_0, m.S2_0
    modes = [md for md in b.modes if abs(md.coeff) > 1e-15]
    vals = np.array([md.coeff * b.psi_at_spot(md, S1, S2) / md.norm for md in modes])
    ks = np.array([md.k_sq for md in modes])
    order = np.argsort(ks)
    k1, k2 = ks[order[0]], ks[order[1]]
    alpha = np.array([m.r - 0.5 * m.sigma1**2, m.r - 0.5 * m.sigma2**2])
    drift = 0.5 * alpha @ np.linalg.inv(m.Sigma) @ alpha
    kappa = m.r + drift + 0.5 * k1
    rows = []
    for T in MATURITIES:
        g = m.gauge_factor(S1, S2, T)
        full = g * np.sum(vals * np.exp(-0.5 * ks * T))
        one = g * vals[order[0]] * np.exp(-0.5 * k1 * T)
        rows.append(dict(T=T, full=float(full), one_mode=float(one),
                         rel=float((one - full) / full)))
    return dict(n_modes=len(modes), lambda1_sq=float(k1), lambda2_sq=float(k2),
                kappa=float(kappa), half_life=float(np.log(2) / kappa),
                parts=dict(discount=m.r, drift=float(drift), barrier=float(0.5 * k1)),
                rows=rows)


def main():
    out = {str(rho): decay_rows(rho) for rho in RHOS}
    path = Path(ARTIFACTS_DIR) / "paper1_decay.json"
    path.write_text(json.dumps(out, indent=1))
    for rho, d in out.items():
        print(f"rho={rho}: kappa={d['kappa']:.3f}/yr, half-life {d['half_life']:.2f} yr, "
              f"parts {d['parts']}")
        for r in d["rows"]:
            print(f"   T={r['T']:4}: full {r['full']:9.5f}  one-mode {r['one_mode']:9.5f}"
                  f"  rel {r['rel']:+.2e}")
    print("wrote", path)


if __name__ == "__main__":
    main()
