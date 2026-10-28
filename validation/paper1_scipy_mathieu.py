"""SciPy's Mathieu routines and the cosh series against the tridiagonal construction
(Remark 3.1)."""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import scipy
from scipy.special import (mathieu_a, mathieu_b, mathieu_modcem1,
                           mathieu_modsem1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twoasset.mathieu_backend import (MathieuAngularTridiagonal,
                                      _locate, radial_value_deriv2)
from twoasset.runlog import build_provenance
from twoasset.solvers._optimal import ARTIFACTS_DIR

OUT = Path(ARTIFACTS_DIR) / "paper1_scipy_mathieu.json"
N_MAX = 8
Q_MAX = 350.0
DQ = 0.05
ORDERS = [(n, p) for n in range(N_MAX + 1) for p in ("even", "odd")
          if not (p == "odd" and n == 0)]


def mu0_of(cfg_name: str) -> float:
    """Barrier coordinate of the whitened Euclidean disc: tanh(mu_0) = b/a."""
    f = tomllib.loads((ROOT / "configs" / f"{cfg_name}.toml").read_text())["financial"]
    s1, s2, rho = f["sigma1"], f["sigma2"], f["rho"]
    d = np.linalg.eigvalsh(np.array([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]))
    return float(np.arctanh(np.sqrt(d[0] / d[1])))


def _scipy_a(n, parity, q):
    return (mathieu_a if parity == "even" else mathieu_b)(n, q)


def _scipy_radial(n, parity, q, mu):
    return (mathieu_modcem1 if parity == "even" else mathieu_modsem1)(n, q, mu)[0]


def branches():
    qs = np.arange(DQ, Q_MAX + DQ / 2, DQ)
    bad = {f"{'ce' if p == 'even' else 'se'}_{n}": [] for n, p in ORDERS}
    for q in qs:
        s = MathieuAngularTridiagonal(q=q, n_max=N_MAX)
        for n, p in ORDERS:
            a_td = s.a_value(*_locate(n, p))
            if abs(_scipy_a(n, p, q) - a_td) > 1e-6 * max(1.0, abs(a_td)):
                bad[f"{'ce' if p == 'even' else 'se'}_{n}"].append(float(q))
    windows = {}
    for name, lst in bad.items():
        if not lst:
            continue
        arr = np.array(lst)
        cut = np.where(np.diff(arr) > 1.5 * DQ)[0]
        windows[name] = [[float(a), float(b)] for a, b in
                         zip(np.r_[arr[0], arr[cut + 1]], np.r_[arr[cut], arr[-1]])]
    # One worked example, the widest window of ce_8.
    q = 35.5
    s = MathieuAngularTridiagonal(q=q, n_max=10)
    example = dict(order="ce_8", q=q, a_scipy=float(mathieu_a(8, q)),
                   a_tridiagonal=s.a_value(*_locate(8, "even")),
                   Mc1_at_mu_0p5_scipy=float(mathieu_modcem1(8, q, 0.5)[0]),
                   Mc1_at_mu_0p5_tridiagonal=float(radial_value_deriv2(s, 8, "even", 0.5, q)[0]))
    n_win = sum(len(v) for v in windows.values())
    print(f"[branches] {n_win} wrong-branch intervals in {len(windows)} of {len(ORDERS)} orders "
          f"(n <= {N_MAX}, q <= {Q_MAX:g}, grid {DQ})", flush=True)
    return dict(dq=DQ, q_max=Q_MAX, n_max=N_MAX, n_orders=len(ORDERS),
                n_orders_affected=len(windows), n_windows=n_win, windows=windows, example=example)


def radial(windows, mu_grids):
    def in_window(n, p, q):
        key = f"{'ce' if p == 'even' else 'se'}_{n}"
        return any(a - 2 * DQ <= q <= b + 2 * DQ for a, b in windows.get(key, []))

    out = {}
    for label, mus in mu_grids.items():
        rows = []
        for q in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 300, 350):
            s = MathieuAngularTridiagonal(q=q, n_max=N_MAX)
            worst = 0.0
            for n, p in ORDERS:
                if in_window(n, p, q):
                    continue
                for mu in mus:
                    t = float(radial_value_deriv2(s, n, p, mu, q)[0])
                    worst = max(worst, abs(_scipy_radial(n, p, q, mu) - t) / max(abs(t), 1e-12))
            rows.append(dict(q=q, max_rel_err=worst))
        out[label] = dict(mu=list(map(float, mus)), rows=rows,
                          max_rel_err=max(r["max_rel_err"] for r in rows))
        print(f"[radial] {label}: max relative error outside the intervals "
              f"{out[label]['max_rel_err']:.1e}", flush=True)
    return out


def cosh(mu0):
    harmonic = {"ce_even": lambda k: 2 * k, "ce_odd": lambda k: 2 * k + 1}
    rows = []
    for q in (5, 20, 50, 100, 200, 350):
        s = MathieuAngularTridiagonal(q=q, n_max=N_MAX)
        worst_err, worst_amp = 0.0, 0.0
        for n in range(N_MAX + 1):
            cname, j = _locate(n, "even")
            A = np.asarray(s.coeffs_natural(cname, j), dtype=float)
            r = np.array([harmonic[cname](k) for k in range(len(A))])
            series = lambda mu: float(np.sum(A * np.cosh(r * mu)))
            # Fix the normalisation where the series is well conditioned.
            k = float(radial_value_deriv2(s, n, "even", 0.1, q)[0]) / series(0.1)
            t = float(radial_value_deriv2(s, n, "even", mu0, q)[0])
            worst_err = max(worst_err, abs(k * series(mu0) - t) / abs(t))
            worst_amp = max(worst_amp, float(np.max(np.abs(A * np.cosh(r * mu0))) / abs(series(mu0))))
        rows.append(dict(q=q, max_rel_err=worst_err, max_term_over_sum=worst_amp))
        print(f"[cosh] q={q:>5g}: largest term / sum {worst_amp:.1e}, relative error {worst_err:.1e}",
              flush=True)
    return dict(mu0=mu0, orders="ce_n, n <= 8", rows=rows)


def main():
    mu0 = mu0_of("default")
    data = dict(scipy_version=scipy.__version__, mu0_default=mu0)
    data["branches"] = branches()
    data["radial"] = radial(data["branches"]["windows"],
                            {"production": np.linspace(0.1, mu0, 5),
                             "thesis_range": np.linspace(0.3, 1.5, 5)})
    data["cosh"] = cosh(mu0)
    data["provenance"] = build_provenance()
    OUT.write_text(json.dumps(data, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
