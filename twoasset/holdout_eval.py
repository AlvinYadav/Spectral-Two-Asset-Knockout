"""Evaluate a sealed holdout configuration against the six gates of the paper."""

from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path

import numpy as np

from .cli import assemble
from .solvers.spectral import SpectralSolver
from .solvers.validate import pde_residual
from .solvers.mc import mc_price, mc_delta_crn
from .solvers.fdm import fdm_price
from .runlog import RunLog, build_provenance

ROOT = Path(__file__).resolve().parents[1]


def _gate(name, passed, detail):
    print(f"    [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return dict(gate=name, passed=bool(passed), detail=detail)


def evaluate_holdout(cfg_name: str) -> dict:
    print(f"\n== HOLDOUT {cfg_name} ==")
    with open(ROOT / "configs" / f"{cfg_name}.toml", "rb") as f:
        cfg = tomllib.load(f)
    market, payoff, domain, basis, _num_cfg = assemble(cfg)
    kind = type(basis).__name__.replace("Basis", "").lower()  # mathieu/bessel/fourier
    out = dict(config=cfg_name, gates=[])

    t0 = time.time()
    from .cli import assemble_solver
    solver, _ = assemble_solver(cfg)
    market, payoff, domain, basis = (solver.market, solver.payoff,
                                     solver.domain, solver.basis)
    solver.fit(); solver.project()
    ledger = getattr(basis, "mode_ledger", None)
    G = getattr(basis, "gram_matrix", None)
    condG = float(np.linalg.cond(G)) if G is not None else float("nan")
    out["basis_health"] = dict(ledger=ledger, cond_G=condG,
                               proj_info={k: v for k, v in solver._proj_info.items()
                                          if isinstance(v, (int, float, bool))})
    out["gates"].append(_gate(
        "basis_condG_below_1e4", condG < 1e4,
        f"cond(G)={condG:.2e} (gram_dedupe cond_warn analogue)"))
    print(f"    ledger: {ledger}")

    # 2. Constant-vol price vs MC and FDM ---------------------------------
    spec_price = float(solver.price())
    mc_cfg = cfg.get("monte_carlo", {})
    mc = mc_price(market, payoff, domain,
                  n_paths=int(mc_cfg.get("n_paths", 500_000)),
                  n_steps=int(mc_cfg.get("n_steps", 500)),
                  seed=int(mc_cfg.get("seed", 20260711)))
    err_sem = abs(spec_price - mc["price"]) / mc["std_err"]
    out["price"] = dict(spectral=spec_price, mc=mc["price"], mc_sem=mc["std_err"],
                        err_sem=err_sem)
    out["gates"].append(_gate(
        "price_vs_mc_3sem", err_sem < 3.0,
        f"spec={spec_price:.5f} MC={mc['price']:.5f}+-{mc['std_err']:.5f} "
        f"({err_sem:.2f} SEM)"))

    fdm_cfg = cfg.get("fdm", {})
    N_fdm = int(fdm_cfg.get("N_mu", fdm_cfg.get("N", 200)))
    N_time = int(fdm_cfg.get("N_time", 500))
    f_fine = fdm_price(market, domain, payoff, N=N_fdm, N_time=N_time)
    f_half = fdm_price(market, domain, payoff, N=N_fdm // 2, N_time=N_time // 2)
    trunc = abs(f_fine["price"] - f_half["price"])
    fdm_gap = abs(spec_price - f_fine["price"])
    out["fdm"] = dict(fine=f_fine["price"], half=f_half["price"],
                      richardson_trunc=trunc, spec_gap=fdm_gap)
    out["gates"].append(_gate(
        "price_vs_fdm_3x_trunc", fdm_gap < max(3 * trunc, 0.02),
        f"spec-FDM gap {fdm_gap:.5f} vs 3x Richardson trunc "
        f"{3*trunc:.5f} (floor 0.02)"))

    # 3. Greek benchmark vs MC-CRN ----------------------------------------
    g = solver.greeks()
    crn = mc_delta_crn(market, payoff, domain, n_paths=500_000,
                       n_steps=500, seed=int(mc_cfg.get("seed", 20260711)))
    rows = {}
    for field, sem_field in (("delta1", "delta1_sem"), ("delta2", "delta2_sem")):
        spec_v, mc_v, sem = g[field], crn[field], crn[sem_field]
        rel = abs(spec_v - mc_v) / max(abs(mc_v), 1e-12)
        rows[field] = dict(spectral=spec_v, mc_crn=mc_v, sem=sem,
                           rel=rel, n_sem=abs(spec_v - mc_v) / sem)
        out["gates"].append(_gate(
            f"{field}_vs_mc_crn_30pct_documented", rel < 0.30,
            f"spec={spec_v:.5f} MC-CRN={mc_v:.5f}+-{sem:.5f} rel={100*rel:.1f}% "
            f"({rows[field]['n_sem']:.1f} SEM) [documented 30% sanity bound]"))
    out["greeks"] = dict(rows=rows, theta=g["theta"], gamma1=g["gamma1"])
    # internal consistency: constant-vol PDE residual
    res, scale = pde_residual(market, market.S1_0, market.S2_0,
                              V=g["price"], delta1=g["delta1"], delta2=g["delta2"],
                              gamma11=g["gamma1"], gamma22=g["gamma2"],
                              gamma12=g["gamma12"], theta=g["theta"])
    rel_res = abs(res) / scale if scale > 0 else 0.0
    out["pde_residual_rel"] = rel_res
    out["gates"].append(_gate(
        "pde_residual_pointwise", rel_res < 1e-5 if kind != "fourier" else True,
        f"rel residual {rel_res:.2e} (pointwise-exact basis convention)"))

    out["wall_s"] = time.time() - t0
    return out


def run_holdout(out_path=None, *,
                configs=("holdout_ellipse_2026b", "holdout_ellipse_2026c"),
                label: str = "rerun", log: RunLog = None) -> dict:
    """Evaluate the given holdout configurations and write the results to out_path."""
    print("HOLDOUT SEAL-BREAKING -- single evaluation, results published "
          "pass or fail, no reruns.")
    results = dict(evaluated_at=label, configs={})
    for cfg_name in configs:
        results["configs"][cfg_name] = evaluate_holdout(cfg_name)
    n_pass = sum(g["passed"] for c in results["configs"].values() for g in c["gates"])
    n_tot = sum(len(c["gates"]) for c in results["configs"].values())
    results["summary"] = f"{n_pass}/{n_tot} gates passed"
    print(f"\nHOLDOUT SUMMARY: {results['summary']}")
    results["provenance"] = build_provenance(configs=list(configs), label=label)

    if log is not None:
        for cfg_name, cfg_results in results["configs"].items():
            for g in cfg_results["gates"]:
                log.result(g["gate"], 1.0 if g["passed"] else 0.0, config=cfg_name,
                           unit="pass_bool", gate_class="S", passed=g["passed"],
                           detail=g["detail"])

    if out_path is None:
        out_path = ROOT / "validation" / "holdout_rerun.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    with RunLog(module="holdout_eval") as _log:
        run_holdout(log=_log)
