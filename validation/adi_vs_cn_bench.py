"""Crank-Nicolson against an ADI splitting on the Euclidean basket (Table 5.1)."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from twoasset.core.market import GBMMarket
from twoasset.core.domain import EuclideanCircle
from twoasset.core.payoff import BasketCall
from twoasset.solvers.fdm import (
    _build_curved_system, _initial_condition_curved, _interp2d, fdm_price,
)


# ---------------------------------------------------------------------------
# Directional operator builder (same SW stencil, split by direction)
# ---------------------------------------------------------------------------
def build_directional(market, domain, N):
    """Return L1, L2, Lc sparse operators + grid metadata."""
    from twoasset.solvers.fdm import _barrier_form
    a, b, c, R2 = _barrier_form(market, domain)
    R = np.sqrt(R2)
    s1, s2, rho, r = market.sigma1, market.sigma2, market.rho, market.r
    half = R  # EuclideanCircle
    dx = 2.0 * half / (N + 1)
    x1_pts = np.array([-half + (i + 1) * dx for i in range(N)])
    x2_pts = np.array([-half + (j + 1) * dx for j in range(N)])

    inside = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            inside[i, j] = (a * x1_pts[i] ** 2 + b * x2_pts[j] ** 2
                            + c * x1_pts[i] * x2_pts[j]) < R2
    flat_idx = -np.ones((N, N), dtype=int)
    cnt = 0
    for i in range(N):
        for j in range(N):
            if inside[i, j]:
                flat_idx[i, j] = cnt
                cnt += 1
    n_unk = cnt

    a11 = 0.5 * s1 ** 2
    a22 = 0.5 * s2 ** 2
    a12 = rho * s1 * s2
    b1 = r - 0.5 * s1 ** 2
    b2 = r - 0.5 * s2 ** 2

    def is_in(ii, jj):
        return 0 <= ii < N and 0 <= jj < N and inside[ii, jj]

    def theta_to_barrier(x1, x2, dx1, dx2):
        A = a * dx1 ** 2 + b * dx2 ** 2 + c * dx1 * dx2
        B = 2 * a * x1 * dx1 + 2 * b * x2 * dx2 + c * (x1 * dx2 + x2 * dx1)
        C = a * x1 ** 2 + b * x2 ** 2 + c * x1 * x2 - R2
        if abs(A) < 1e-30:
            return 1.0
        disc = B * B - 4 * A * C
        if disc < 0:
            return 1.0
        sd = np.sqrt(disc)
        roots = [(-B + sd) / (2 * A), (-B - sd) / (2 * A)]
        valid = [t for t in roots if 1e-10 < t <= 1.0 + 1e-8]
        if not valid:
            return 1.0
        return max(min(min(valid), 1.0), 1e-6)

    r1, c1, v1 = [], [], []
    r2, c2, v2 = [], [], []
    rc, cc, vc = [], [], []

    for i in range(N):
        for j in range(N):
            if not inside[i, j]:
                continue
            k = flat_idx[i, j]
            x1c, x2c = x1_pts[i], x2_pts[j]

            # ---- x1 direction (L1) ----
            diag1 = -r / 2.0
            ip_in = is_in(i + 1, j); im_in = is_in(i - 1, j)
            hp = dx if ip_in else theta_to_barrier(x1c, x2c, dx, 0) * dx
            hm = dx if im_in else theta_to_barrier(x1c, x2c, -dx, 0) * dx
            cp2 = a11 / (hp * (hp + hm) / 2); cm2 = a11 / (hm * (hp + hm) / 2)
            diag1 += -(cp2 + cm2)
            if ip_in: r1.append(k); c1.append(flat_idx[i + 1, j]); v1.append(cp2)
            if im_in: r1.append(k); c1.append(flat_idx[i - 1, j]); v1.append(cm2)
            den = hp * hm * (hp + hm)
            cp1 = b1 * hm ** 2 / den; cm1 = -b1 * hp ** 2 / den
            diag1 += b1 * (hp ** 2 - hm ** 2) / den
            if ip_in: r1.append(k); c1.append(flat_idx[i + 1, j]); v1.append(cp1)
            if im_in: r1.append(k); c1.append(flat_idx[i - 1, j]); v1.append(cm1)
            r1.append(k); c1.append(k); v1.append(diag1)

            # ---- x2 direction (L2) ----
            diag2 = -r / 2.0
            jp_in = is_in(i, j + 1); jm_in = is_in(i, j - 1)
            hp2 = dx if jp_in else theta_to_barrier(x1c, x2c, 0, dx) * dx
            hm2 = dx if jm_in else theta_to_barrier(x1c, x2c, 0, -dx) * dx
            cp2b = a22 / (hp2 * (hp2 + hm2) / 2); cm2b = a22 / (hm2 * (hp2 + hm2) / 2)
            diag2 += -(cp2b + cm2b)
            if jp_in: r2.append(k); c2.append(flat_idx[i, j + 1]); v2.append(cp2b)
            if jm_in: r2.append(k); c2.append(flat_idx[i, j - 1]); v2.append(cm2b)
            den2 = hp2 * hm2 * (hp2 + hm2)
            cp1b = b2 * hm2 ** 2 / den2; cm1b = -b2 * hp2 ** 2 / den2
            diag2 += b2 * (hp2 ** 2 - hm2 ** 2) / den2
            if jp_in: r2.append(k); c2.append(flat_idx[i, j + 1]); v2.append(cp1b)
            if jm_in: r2.append(k); c2.append(flat_idx[i, j - 1]); v2.append(cm1b)
            r2.append(k); c2.append(k); v2.append(diag2)

            # ---- cross (Lc) ----
            c12 = a12 / (4.0 * dx ** 2)
            if is_in(i + 1, j + 1): rc.append(k); cc.append(flat_idx[i + 1, j + 1]); vc.append(c12)
            if is_in(i + 1, j - 1): rc.append(k); cc.append(flat_idx[i + 1, j - 1]); vc.append(-c12)
            if is_in(i - 1, j + 1): rc.append(k); cc.append(flat_idx[i - 1, j + 1]); vc.append(-c12)
            if is_in(i - 1, j - 1): rc.append(k); cc.append(flat_idx[i - 1, j - 1]); vc.append(c12)

    L1 = sparse.csr_matrix((v1, (r1, c1)), shape=(n_unk, n_unk))
    L2 = sparse.csr_matrix((v2, (r2, c2)), shape=(n_unk, n_unk))
    Lc = sparse.csr_matrix((vc, (rc, cc)), shape=(n_unk, n_unk))
    return dict(L1=L1, L2=L2, Lc=Lc, x1=x1_pts, x2=x2_pts, inside=inside,
                flat_idx=flat_idx, n_unk=n_unk, dx=dx)


def solve_cn(market, domain, payoff, N, N_time, rannacher_steps=4):
    """Monolithic Crank-Nicolson + Rannacher (mirrors production fdm_price)."""
    x1, x2, ins, fidx, n_unk, L, dx = _build_curved_system(market, domain, N)
    dt = market.T / N_time
    I = sparse.eye(n_unk, format="csr")
    u = _initial_condition_curved(market, payoff, x1, x2, ins, fidx, n_unk)

    t0 = time.perf_counter()
    n_cn = N_time
    dt_half = dt / 2.0
    ie = splu((I - dt_half * L).tocsc())
    for _ in range(2 * rannacher_steps):
        u = ie.solve(u)
    n_cn -= rannacher_steps
    solver = splu((I - 0.5 * dt * L).tocsc())
    t_factor = time.perf_counter() - t0

    B = I + 0.5 * dt * L
    t1 = time.perf_counter()
    for _ in range(n_cn):
        u = solver.solve(B @ u)
    t_step = time.perf_counter() - t1

    x1t = float(np.log(market.S1_0 / market.S1_0))
    x2t = float(np.log(market.S2_0 / market.S2_0))
    price = _interp2d(u, x1t, x2t, x1, x2, ins, fidx, N)
    return dict(price=price, t_factor=t_factor, t_step=t_step,
                t_total=t_factor + t_step, n_unk=n_unk)


def solve_adi_pr(market, domain, payoff, N, N_time):
    """Peaceman-Rachford ADI without damping."""
    d = build_directional(market, domain, N)
    L1, L2, Lc = d["L1"], d["L2"], d["Lc"]
    n_unk = d["n_unk"]
    dt = market.T / N_time
    I = sparse.eye(n_unk, format="csr")
    u = _initial_condition_curved(market, payoff, d["x1"], d["x2"],
                                  d["inside"], d["flat_idx"], n_unk)
    t0 = time.perf_counter()
    A1 = splu((I - 0.5 * dt * L1).tocsc())
    A2 = splu((I - 0.5 * dt * L2).tocsc())
    t_factor = time.perf_counter() - t0
    R2op = I + 0.5 * dt * (L2 + Lc)
    R1op = I + 0.5 * dt * (L1 + Lc)
    t1 = time.perf_counter()
    for _ in range(N_time):
        ustar = A1.solve(R2op @ u)
        u = A2.solve(R1op @ ustar)
    t_step = time.perf_counter() - t1
    price = _interp2d(u, 0.0, 0.0, d["x1"], d["x2"], d["inside"],
                      d["flat_idx"], N)
    return dict(price=price, t_factor=t_factor, t_step=t_step,
                t_total=t_factor + t_step, n_unk=n_unk)


def solve_adi(market, domain, payoff, N, N_time, rannacher_steps=4):
    """Douglas ADI with a theta=1 Rannacher-style warm-up for kink damping."""
    d = build_directional(market, domain, N)
    L1, L2, Lc = d["L1"], d["L2"], d["Lc"]
    Lfull = (L1 + L2 + Lc).tocsr()
    n_unk = d["n_unk"]
    dt = market.T / N_time
    I = sparse.eye(n_unk, format="csr")
    u = _initial_condition_curved(market, payoff, d["x1"], d["x2"],
                                  d["inside"], d["flat_idx"], n_unk)

    t0 = time.perf_counter()
    # theta = 1 (warm-up) and theta = 1/2 (bulk): factor each direction once.
    A1_be = splu((I - 1.0 * dt * L1).tocsc())
    A2_be = splu((I - 1.0 * dt * L2).tocsc())
    A1_cn = splu((I - 0.5 * dt * L1).tocsc())
    A2_cn = splu((I - 0.5 * dt * L2).tocsc())
    t_factor = time.perf_counter() - t0

    def douglas_step(u, theta, A1, A2):
        Y0 = u + dt * (Lfull @ u)
        Y1 = A1.solve(Y0 - theta * dt * (L1 @ u))
        Y2 = A2.solve(Y1 - theta * dt * (L2 @ u))
        return Y2

    t1 = time.perf_counter()
    for _ in range(rannacher_steps):
        u = douglas_step(u, 1.0, A1_be, A2_be)
    for _ in range(N_time - rannacher_steps):
        u = douglas_step(u, 0.5, A1_cn, A2_cn)
    t_step = time.perf_counter() - t1

    price = _interp2d(u, 0.0, 0.0, d["x1"], d["x2"], d["inside"],
                      d["flat_idx"], N)
    return dict(price=price, t_factor=t_factor, t_step=t_step,
                t_total=t_factor + t_step, n_unk=n_unk)


def verify_split(market, domain, N):
    """Confirm L1+L2+Lc equals the production monolithic operator."""
    _, _, _, _, _, L, _ = _build_curved_system(market, domain, N)
    d = build_directional(market, domain, N)
    Lsum = (d["L1"] + d["L2"] + d["Lc"]).tocsr()
    diff = (L - Lsum)
    return abs(diff).max() if diff.nnz else 0.0


def main():
    market = GBMMarket(S1_0=100.0, S2_0=100.0, sigma1=0.20, sigma2=0.30,
                       rho=0.50, r=0.05, T=1.0)
    domain = EuclideanCircle(R=0.5)
    payoff = BasketCall(K=180.0, w1=1.0, w2=1.0)
    N_time = 500

    print("=" * 78)
    print("ADI vs Crank-Nicolson FDM benchmark")
    print("market: S=(100,100) sig=(.20,.30) rho=.5 r=.05 T=1  barrier R=0.5")
    print(f"payoff: BasketCall K=180   N_time={N_time}")
    print("=" * 78)

    # Sanity: directional split reproduces the monolithic operator
    sp = verify_split(market, domain, 60)
    print(f"\n[split check] max|L - (L1+L2+Lc)| at N=60 : {sp:.2e}  "
          f"({'OK' if sp < 1e-9 else 'MISMATCH'})")

    Ns = [80, 120, 160, 240, 320]
    rows = []
    print(f"\nCN = monolithic Crank-Nicolson + Rannacher (what the CODE does)")
    print(f"ADI = Douglas 1961 / in't Hout 2010 + theta=1 Rannacher warm-up (stable)")
    print(f"PR  = naive Peaceman-Rachford (no damping)\n")
    print(f"{'N':>5} {'n_unk':>7} | {'CN price':>11} {'ADI price':>11} "
          f"{'PR price':>13} | {'CN-ADI Δ':>9} | {'CN fac':>7} {'CN tot':>8} | "
          f"{'ADI fac':>7} {'ADI tot':>8} | {'speed':>6}")
    print("-" * 122)
    for N in Ns:
        cn = solve_cn(market, domain, payoff, N, N_time)
        adi = solve_adi(market, domain, payoff, N, N_time)
        pr = solve_adi_pr(market, domain, payoff, N, N_time)
        dprice = abs(cn["price"] - adi["price"])
        speed = cn["t_total"] / adi["t_total"]
        pr_str = (f"{pr['price']:>13.4f}" if abs(pr["price"]) < 1e6
                  else f"{'BLEW UP':>13}")
        rows.append(dict(N=N, n_unk=cn["n_unk"], cn=cn, adi=adi, pr=pr,
                         dprice=dprice, speed=speed))
        print(f"{N:>5} {cn['n_unk']:>7} | {cn['price']:>11.6f} "
              f"{adi['price']:>11.6f} {pr_str} | {dprice:>9.2e} | "
              f"{cn['t_factor']:>7.3f} {cn['t_total']:>8.3f} | "
              f"{adi['t_factor']:>7.3f} {adi['t_total']:>8.3f} | "
              f"{speed:>5.2f}x")

    # persist
    out = dict(
        market=dict(S1_0=100, S2_0=100, sigma1=0.2, sigma2=0.3, rho=0.5,
                    r=0.05, T=1.0, R=0.5, K=180.0),
        N_time=N_time,
        split_check_maxerr=sp,
        rows=[dict(N=r["N"], n_unk=r["n_unk"],
                   cn_price=r["cn"]["price"], adi_price=r["adi"]["price"],
                   pr_naive_price=r["pr"]["price"],
                   abs_price_diff_cn_adi=r["dprice"],
                   cn_t_factor=r["cn"]["t_factor"], cn_t_step=r["cn"]["t_step"],
                   cn_t_total=r["cn"]["t_total"],
                   adi_t_factor=r["adi"]["t_factor"], adi_t_step=r["adi"]["t_step"],
                   adi_t_total=r["adi"]["t_total"],
                   cn_over_adi_speedup=r["speed"]) for r in rows],
    )
    out_path = Path(__file__).resolve().parent / "adi_vs_cn_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
