"""Compare projection measures for the Mathieu basis (Table 6.6): unweighted
collocation, Jacobian-weighted least squares, and direct Galerkin integration."""

from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path

import numpy as np

from .core.market import GBMMarket
from .core.payoff import BasketCall
from .core.domain import EuclideanCircle
from .bases.mathieu import MathieuBasis
from .solvers.spectral import SpectralSolver
from .runlog import RunLog, build_provenance

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "validation" / "projection_comparison.json"

ARMS = ("collocation", "weighted", "galerkin")
CALIBRATION_CONFIGS = ("default", "tight_barrier")
HOLDOUT_CONFIG = "holdout_ellipse"
RUNGS = [(8, 5, 350.0), (8, 6, 400.0)]
SMOOTHING_GRID = (0.0, 0.005, 0.03, 0.05)
Q_MAX_CAP = 1600.0

REFS = {
    "default":         dict(fdm=18.9946646518, mc=18.99347404,  sem=0.00548114),
    "tight_barrier":   dict(fdm=4.4385199305,  mc=4.435500251,  sem=0.00242578),
    "holdout_ellipse": dict(fdm=37.1111250,    mc=37.10702467,  sem=0.018685),
}
FDM_REL_TOL = {"default": 5e-4, "tight_barrier": 5e-4, "holdout_ellipse": 1e-3}
INTER_RUNG_REL_TOL = 5e-4
MC_SEM_MULT = 3.0


def _scan_coverage(basis: MathieuBasis) -> dict:
    """Report whether every angular order and parity found the requested number of
    radial roots."""
    counts = {
        (n, parity): 0
        for n, parity in (
            [(n, "even") for n in range(basis.N_angular_max + 1)]
            + [(n, "odd") for n in range(1, basis.N_angular_max + 1)]
        )
    }
    for md in basis.modes:
        key = (int(md.meta["n"]), md.meta["parity"])
        counts[key] = counts.get(key, 0) + 1

    families = [
        dict(
            n=n,
            parity=parity,
            found=count,
            requested=basis.M_radial_max,
            shortfall=max(0, basis.M_radial_max - count),
        )
        for (n, parity), count in sorted(counts.items())
    ]
    incomplete = [row for row in families if row["shortfall"] > 0]
    requested_total = basis.M_radial_max * len(families)
    return dict(
        complete=not incomplete,
        requested_total=requested_total,
        found_total=len(basis.modes),
        total_shortfall=requested_total - len(basis.modes),
        incomplete_families=incomplete,
        families=families,
        max_q_found=max((float(md.meta["q"]) for md in basis.modes), default=None),
    )


def _build_solver(cfg_name, n_ang, m_rad, q_max):
    with open(ROOT / "configs" / f"{cfg_name}.toml", "rb") as f:
        cfg = tomllib.load(f)
    market = GBMMarket.from_config(cfg)
    fin = cfg["financial"]
    payoff = BasketCall(K=fin["K"], w1=fin.get("w1", 1.0),
                        w2=fin.get("w2", 1.0))
    domain = EuclideanCircle(R=cfg["barrier"]["R"])
    num = cfg["numerical"]
    basis = MathieuBasis(
        N_trunc=int(num["N_trunc"]),
        N_angular_max=n_ang, M_radial_max=m_rad, q_max=q_max,
        q_scan_points=int(15 * q_max),
        N_mu_quad=max(int(num["N_mu_quad"]), 10 * m_rad),
        N_nu_quad=max(int(num["N_nu_quad"]), 8 * n_ang),
        backend="tridiagonal",
    )
    return SpectralSolver(market=market, payoff=payoff, domain=domain,
                          basis=basis, smooth=True, smooth_tau_frac=0.005)


def _fit_complete_rung(cfg_name, n_ang, m_rad, q_max0):
    """Fit the basis, raising q_max until every requested root is found."""
    q_max = float(q_max0)
    while True:
        solver = _build_solver(cfg_name, n_ang, m_rad, q_max)
        t0 = time.perf_counter()
        solver.fit()
        fit_s = time.perf_counter() - t0
        coverage = _scan_coverage(solver.basis)
        if coverage["complete"]:
            return solver, coverage, q_max, fit_s
        if q_max >= Q_MAX_CAP:
            raise RuntimeError(
                f"{cfg_name} {n_ang}x{m_rad}: rung incomplete at q_max="
                f"{q_max} (found {coverage['found_total']}/"
                f"{coverage['requested_total']}); cap {Q_MAX_CAP} reached"
            )
        q_max = float(round(q_max * 1.5))


def _j_gram_diagnostic(basis):
    """Largest off-diagonal entry of the Jacobian-weighted Gram matrix."""
    mu_nodes, mu_w = basis._mu_quad
    nu_nodes = basis._nu_quad
    dnu = basis._dnu
    n_modes = len(basis.modes)
    R = np.empty((n_modes, len(mu_nodes)))
    A = np.empty((n_modes, len(nu_nodes)))
    norms = np.array([md.norm if md.norm > 1e-15 else 1.0
                      for md in basis.modes])
    for k, md in enumerate(basis.modes):
        n, q, parity = md.meta["n"], md.meta["q"], md.meta["parity"]
        Rv, _, _, Av, _, _ = basis._separated_value_deriv2(
            n, q, parity, mu_nodes, nu_nodes)
        R[k] = Rv
        A[k] = Av
    sinh2 = np.sinh(mu_nodes) ** 2
    sin2 = np.sin(nu_nodes) ** 2
    Mr_plain = (R * mu_w) @ R.T
    Mr_sinh = (R * (mu_w * sinh2)) @ R.T
    Ma_plain = dnu * (A @ A.T)
    Ma_sin = dnu * ((A * sin2) @ A.T)
    G = basis.c_focal ** 2 * (Mr_sinh * Ma_plain + Mr_plain * Ma_sin)
    Ghat = G / np.outer(norms, norms)
    off = Ghat - np.diag(np.diag(Ghat))
    return dict(
        max_offdiag=float(np.max(np.abs(off))),
        max_diag_err=float(np.max(np.abs(np.diag(Ghat) - 1.0))),
    )


def _run_rung_cells(solver, cfg_name, n_ang, m_rad, *,
                    arms=ARMS, smoothing_grid=SMOOTHING_GRID):
    """Price every combination of smoothing and projection method on one fitted
    basis."""
    rows, consistency = [], []
    for frac in smoothing_grid:
        arm_data = {}
        for arm in arms:
            solver.projection_method = arm
            solver.smooth = frac > 0
            solver.smooth_tau_frac = max(frac, 1e-12)
            t0 = time.perf_counter()
            info = solver.project(reuse_design=True)
            price = float(solver.price())
            proj_s = time.perf_counter() - t0
            coeffs = np.array([md.coeff for md in solver.basis.modes])
            arm_data[arm] = (price, coeffs)
            rows.append(dict(
                config=cfg_name, N_angular_max=n_ang, M_radial_max=m_rad,
                smooth_tau_frac=frac, arm=arm, price=price,
                projection_cond=float(info["cond"]),
                n_active=int(info["n_active"]), wall_s=proj_s,
            ))
        if "weighted" in arm_data and "galerkin" in arm_data:
            p1, c1 = arm_data["weighted"]
            p2, c2 = arm_data["galerkin"]
            denom = max(float(np.linalg.norm(c2)), 1e-300)
            consistency.append(dict(
                config=cfg_name, N_angular_max=n_ang, M_radial_max=m_rad,
                smooth_tau_frac=frac,
                coeff_rel_diff=float(np.linalg.norm(c1 - c2)) / denom,
                price_abs_diff=abs(p1 - p2),
            ))
    return rows, consistency


def _gates_for(rows, cfg_name, arm, frac):
    """Accuracy and truncation-stability checks for one configuration, method and
    smoothing."""
    ref = REFS[cfg_name]
    cell = {}
    for r in rows:
        if (r["config"] == cfg_name and r["arm"] == arm
                and r["smooth_tau_frac"] == frac):
            cell[f"{r['N_angular_max']}x{r['M_radial_max']}"] = r["price"]
    if len(cell) != 2:
        return None
    prices = list(cell.values())
    fdm_rel = [abs(p - ref["fdm"]) / ref["fdm"] for p in prices]
    mc_dev = [abs(p - ref["mc"]) / ref["sem"] for p in prices]
    inter = abs(prices[0] - prices[1]) / ref["fdm"]
    return dict(
        prices=cell,
        worst_fdm_rel=max(fdm_rel),
        worst_mc_sems=max(mc_dev),
        inter_rung_rel=inter,
        accuracy_pass=(max(fdm_rel) <= FDM_REL_TOL[cfg_name]
                       and max(mc_dev) <= MC_SEM_MULT),
        inter_rung_pass=inter <= INTER_RUNG_REL_TOL,
    )


def run_study(out_path=OUT_PATH, *, log: RunLog = None) -> dict:
    own_log = log is None
    if own_log:
        log = RunLog(module="projection_comparison")
    log.params(rungs=RUNGS, smoothing_grid=SMOOTHING_GRID, arms=ARMS,
               refs=REFS, fdm_rel_tol=FDM_REL_TOL,
               inter_rung_rel_tol=INTER_RUNG_REL_TOL,
               mc_sem_mult=MC_SEM_MULT, backend="tridiagonal")

    out = dict(status="running", rungs=RUNGS,
               smoothing_grid=list(SMOOTHING_GRID),
               rows=[], consistency=[], j_gram=[], fits=[],
               provenance=build_provenance(rungs=RUNGS))

    def checkpoint():
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)

    # ── Production contracts: every method and smoothing ──────────
    for cfg_name in CALIBRATION_CONFIGS:
        for n_ang, m_rad, q0 in RUNGS:
            print(f"[{cfg_name} {n_ang}x{m_rad}] fitting ...", flush=True)
            solver, coverage, q_max, fit_s = _fit_complete_rung(
                cfg_name, n_ang, m_rad, q0)
            out["fits"].append(dict(
                config=cfg_name, N_angular_max=n_ang, M_radial_max=m_rad,
                q_max_requested=q0, q_max_used=q_max,
                n_modes=len(solver.basis.modes), fit_s=fit_s,
                scan_coverage=coverage))
            jg = _j_gram_diagnostic(solver.basis)
            jg.update(config=cfg_name, N_angular_max=n_ang,
                      M_radial_max=m_rad)
            out["j_gram"].append(jg)
            log.result("j_gram_max_offdiag", jg["max_offdiag"],
                       config=cfg_name, gate_class="D",
                       shape=f"{n_ang}x{m_rad}")
            rows, cons = _run_rung_cells(solver, cfg_name, n_ang, m_rad)
            out["rows"].extend(rows)
            out["consistency"].extend(cons)
            for c in cons:
                log.result("c1_c2_coeff_rel_diff", c["coeff_rel_diff"],
                           config=cfg_name, gate_class="D",
                           shape=f"{n_ang}x{m_rad}",
                           smooth_tau_frac=c["smooth_tau_frac"])
            checkpoint()

    # ── Accuracy checks on the production contracts ───────────────
    verdicts = {}
    for arm in ARMS:
        per_frac = {}
        for frac in SMOOTHING_GRID:
            gates = {c: _gates_for(out["rows"], c, arm, frac)
                     for c in CALIBRATION_CONFIGS}
            ok = all(g is not None and g["accuracy_pass"]
                     and g["inter_rung_pass"] for g in gates.values())
            per_frac[frac] = dict(passes=ok, detail=gates)
        passing = [f for f in SMOOTHING_GRID if per_frac[f]["passes"]]
        verdicts[arm] = dict(
            passing_smoothings=passing,
            selected_smoothing=passing[0] if passing else None,
            per_smoothing=per_frac,
        )
    out["calibration_verdicts"] = verdicts

    primary = None
    for arm in ("galerkin", "weighted"):
        if verdicts[arm]["selected_smoothing"] is not None:
            primary = arm
            break
    out["primary_arm"] = primary
    checkpoint()

    # ── Holdout A with the selected method ─────────────────────────
    if primary is None:
        out["status"] = "failed_calibration_gates"
        out["holdout"] = None
        checkpoint()
        if own_log:
            log.close()
        print("No experimental arm passed the calibration gates; the "
              "holdout was NOT run (stop rule §7.1).")
        return out

    sel_frac = verdicts[primary]["selected_smoothing"]
    out["holdout"] = dict(arm=primary, smooth_tau_frac=sel_frac, rows=[],
                          consistency=[], j_gram=[], fits=[])
    for n_ang, m_rad, q0 in RUNGS:
        print(f"[{HOLDOUT_CONFIG} {n_ang}x{m_rad}] fitting ...", flush=True)
        solver, coverage, q_max, fit_s = _fit_complete_rung(
            HOLDOUT_CONFIG, n_ang, m_rad, q0)
        out["holdout"]["fits"].append(dict(
            config=HOLDOUT_CONFIG, N_angular_max=n_ang, M_radial_max=m_rad,
            q_max_requested=q0, q_max_used=q_max,
            n_modes=len(solver.basis.modes), fit_s=fit_s,
            scan_coverage=coverage))
        jg = _j_gram_diagnostic(solver.basis)
        jg.update(config=HOLDOUT_CONFIG, N_angular_max=n_ang,
                  M_radial_max=m_rad)
        out["holdout"]["j_gram"].append(jg)
        rows, cons = _run_rung_cells(
            solver, HOLDOUT_CONFIG, n_ang, m_rad,
            arms=("weighted", "galerkin"), smoothing_grid=(sel_frac,))
        out["holdout"]["rows"].extend(rows)
        out["holdout"]["consistency"].extend(cons)
        checkpoint()

    hg = _gates_for(out["holdout"]["rows"], HOLDOUT_CONFIG, primary,
                    sel_frac)
    out["holdout"]["gates"] = hg
    log.result("holdout_worst_fdm_rel", hg["worst_fdm_rel"],
               config=HOLDOUT_CONFIG, gate_class="S",
               threshold=FDM_REL_TOL[HOLDOUT_CONFIG],
               passed=hg["accuracy_pass"], arm=primary,
               smooth_tau_frac=sel_frac)
    log.result("holdout_inter_rung_rel", hg["inter_rung_rel"],
               config=HOLDOUT_CONFIG, gate_class="S",
               threshold=INTER_RUNG_REL_TOL, passed=hg["inter_rung_pass"],
               arm=primary, smooth_tau_frac=sel_frac)
    out["status"] = ("closed" if hg["accuracy_pass"] and hg["inter_rung_pass"]
                     else "failed_holdout_gates")
    checkpoint()
    if own_log:
        log.close()

    # ── Console summary ─────────────────────────────────────────────
    print(f"\nprimary arm: {primary}   selected smoothing: {sel_frac}")
    for arm in ARMS:
        v = verdicts[arm]
        print(f"  {arm:<12} passing smoothings: {v['passing_smoothings']}")
    print(f"holdout gates: accuracy_pass={hg['accuracy_pass']} "
          f"(worst FDM rel {hg['worst_fdm_rel']:.5%}, "
          f"worst MC dev {hg['worst_mc_sems']:.2f} SEM), "
          f"inter_rung_pass={hg['inter_rung_pass']} "
          f"({hg['inter_rung_rel']:.5%})")
    print(f"STATUS: {out['status']}")
    return out


if __name__ == "__main__":
    run_study()
