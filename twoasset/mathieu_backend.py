"""Mathieu functions from the three-term recurrence, posed as symmetric tridiagonal
eigenproblems."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import jv


def _truncation_K(n_max: int) -> int:
    return max(2 * n_max + 16, 32)


_BESSEL_CACHE_MAX = 64


def _memo_store(cache: dict, key, value) -> None:
    """Insert into a bounded memo, clearing it first when full."""
    if len(cache) >= _BESSEL_CACHE_MAX:
        cache.clear()
    cache[key] = value


def _class_matrix(class_name: str, q: float, K: int):
    """Diagonal and off-diagonal of the symmetric tridiagonal matrix for one symmetry
    class."""
    k = np.arange(K)
    if class_name == "ce_even":            # ce_{2m}: A_0, A_2, A_4, ...
        diag = (2.0 * k) ** 2
        offdiag = np.full(K - 1, q)
        offdiag[0] = np.sqrt(2.0) * q
    elif class_name == "ce_odd":           # ce_{2m+1}: A_1, A_3, A_5, ...
        diag = (2.0 * k + 1.0) ** 2
        diag[0] = 1.0 + q
        offdiag = np.full(K - 1, q)
    elif class_name == "se_odd":           # se_{2m+1}: B_1, B_3, B_5, ...
        diag = (2.0 * k + 1.0) ** 2
        diag[0] = 1.0 - q
        offdiag = np.full(K - 1, q)
    elif class_name == "se_even":          # se_{2m+2}: B_2, B_4, B_6, ...
        diag = (2.0 * k + 2.0) ** 2
        offdiag = np.full(K - 1, q)
    else:
        raise ValueError(f"unknown symmetry class {class_name!r}")
    return diag, offdiag


CLASSES = ("ce_even", "ce_odd", "se_odd", "se_even")
# k-index -> actual Fourier index (the "k" that appears in cos(k*nu)/sin(k*nu))
FOURIER_INDEX = {
    "ce_even": lambda j: 2 * j,
    "ce_odd":  lambda j: 2 * j + 1,
    "se_odd":  lambda j: 2 * j + 1,
    "se_even": lambda j: 2 * j + 2,
}
# eigenvalue-index j -> Mathieu order n
ORDER_OF = {
    "ce_even": lambda j: 2 * j,
    "ce_odd":  lambda j: 2 * j + 1,
    "se_odd":  lambda j: 2 * j + 1,
    "se_even": lambda j: 2 * j + 2,
}


def _fix_sign(class_name: str, coeffs: np.ndarray) -> np.ndarray:
    """Fix the sign of each eigenvector so that ce_n(0) > 0 and se_n'(0) > 0."""
    K = coeffs.shape[0]
    fourier_idx = np.array([FOURIER_INDEX[class_name](j) for j in range(K)])
    if class_name in ("ce_even", "ce_odd"):
        s = np.sum(coeffs, axis=0)
    else:
        s = np.sum(coeffs * fourier_idx[:, None], axis=0)
    flip = s < 0
    out = coeffs.copy()
    out[:, flip] *= -1
    return out


class MathieuAngularTridiagonal:
    """Angular Mathieu functions of all four symmetry classes at a fixed q, up to
    order n_max."""

    def __init__(self, q: float, n_max: int, K: int = None):
        self.q = float(q)
        self.n_max = n_max
        self.K = K if K is not None else _truncation_K(n_max)
        self._bessel_cache = {}
        self._solve()

    def _solve(self):
        self.a_values = {}
        self.coeffs = {}
        while True:
            ok = True
            for cname in CLASSES:
                diag, offdiag = _class_matrix(cname, self.q, self.K)
                a_vals, vecs = eigh_tridiagonal(diag, offdiag)
                vecs = _fix_sign(cname, vecs)
                self.a_values[cname] = a_vals
                self.coeffs[cname] = vecs
                n_check = min(self.n_max // 2 + 2, vecs.shape[1])
                sub = vecs[:, :n_check]
                max_c = np.max(np.abs(sub), axis=0)
                last_c = np.abs(sub[-1, :])
                if not np.all(last_c < 1e-15 * np.maximum(max_c, 1e-300)):
                    ok = False
            if ok:
                break
            self.K *= 2
            if self.K > 4096:
                raise RuntimeError(
                    f"D1 truncation did not converge by K=4096 at "
                    f"q={self.q}, n_max={self.n_max}"
                )

    def coeffs_natural(self, class_name: str, j: int) -> np.ndarray:
        """The natural A_k/B_k coefficient vector (length K) for eigenvalue-index j
        of `class_name`, un-doing the D=diag(sqrt2,1,1,..) symmetrisation for
        ce_even."""
        v = self.coeffs[class_name][:, j].copy()
        if class_name == "ce_even":
            v = v.copy()
            v[0] /= np.sqrt(2.0)
        return v

    def a_value(self, class_name: str, j: int) -> float:
        return float(self.a_values[class_name][j])


def _locate(n: int, parity: str):
    """Mathieu order n + parity ('even'=ce, 'odd'=se) -> (class_name, j)."""
    if parity == "even":
        if n % 2 == 0:
            return "ce_even", n // 2
        return "ce_odd", (n - 1) // 2
    else:
        if n == 0:
            raise ValueError("se_0 does not exist")
        if n % 2 == 1:
            return "se_odd", (n - 1) // 2
        return "se_even", n // 2 - 1


def angular_value_deriv2(solver: MathieuAngularTridiagonal, n: int, parity: str,
                         nu) -> tuple:
    """ce_n(nu) or se_n(nu) and its first two nu-derivatives, as a direct
    trigonometric sum of the natural coefficients."""
    cname, j = _locate(n, parity)
    coeffs = solver.coeffs_natural(cname, j)
    K = len(coeffs)
    fourier_idx = np.array([FOURIER_INDEX[cname](k) for k in range(K)])

    nu_arr = np.atleast_1d(np.asarray(nu, dtype=float))
    kn = fourier_idx[:, None] * nu_arr[None, :]        # (K, n_pts)

    if parity == "even":
        trig = np.cos(kn)
        dtrig = -fourier_idx[:, None] * np.sin(kn)
    else:
        trig = np.sin(kn)
        dtrig = fourier_idx[:, None] * np.cos(kn)
    d2trig = -(fourier_idx[:, None] ** 2) * trig

    val = coeffs @ trig
    dval = coeffs @ dtrig
    d2val = coeffs @ d2trig

    scalar_in = np.isscalar(nu) or np.asarray(nu).ndim == 0
    if scalar_in:
        return float(val[0]), float(dval[0]), float(d2val[0])
    shape = np.asarray(nu).shape
    return val.reshape(shape), dval.reshape(shape), d2val.reshape(shape)


def gate_G1_angular_ode_residual(q: float, n_max: int, *, n_nu: int = 400) -> dict:
    """||ce'' + (a - 2q cos2nu) ce||_inf / ||ce||_inf < 1e-11 (and se), sampled on a
    fine nu grid, for every n in [0, n_max] (se from n=1)."""
    solver = MathieuAngularTridiagonal(q=q, n_max=n_max)
    nu = np.linspace(0, 2 * np.pi, n_nu, endpoint=False)
    rows = []
    for n in range(n_max + 1):
        for parity in ("even", "odd"):
            if parity == "odd" and n == 0:
                continue
            cname, j = _locate(n, parity)
            a = solver.a_value(cname, j)
            val, _, d2val = angular_value_deriv2(solver, n, parity, nu)
            residual = d2val + (a - 2 * q * np.cos(2 * nu)) * val
            denom = max(np.max(np.abs(val)), 1e-300)
            rel = float(np.max(np.abs(residual)) / denom)
            rows.append(dict(n=n, parity=parity, q=q, residual_rel=rel))
    worst = max(rows, key=lambda r: r["residual_rel"])
    passed = worst["residual_rel"] < 1e-11
    return dict(gate="G1", passed=passed, threshold=1e-11, rows=rows, worst=worst)


def gate_G2_angular_orthonormality(q: float, n_max: int, *, n_nu: int = 2000) -> dict:
    """integral_0^2pi ce_m ce_n dnu = pi * delta_mn to 1e-12 by quadrature, within
    each class (cross-class/cross-parity orthogonality is automatic by symmetry
    and not separately gated)."""
    solver = MathieuAngularTridiagonal(q=q, n_max=n_max)
    nu = np.linspace(0, 2 * np.pi, n_nu, endpoint=False)
    dnu = 2 * np.pi / n_nu
    rows = []
    for parity in ("even", "odd"):
        orders = [n for n in range(0 if parity == "even" else 1, n_max + 1)
                 if (parity == "even") or (n >= 1)]
        vals = {}
        for n in orders:
            v, _, _ = angular_value_deriv2(solver, n, parity, nu)
            vals[n] = v
        for i, ni in enumerate(orders):
            for nj in orders[i:]:
                ip = float(np.sum(vals[ni] * vals[nj]) * dnu)
                target = np.pi if ni == nj else 0.0
                err = abs(ip - target)
                rows.append(dict(n_i=ni, n_j=nj, parity=parity,
                                 inner_product=ip, target=target, abs_err=err))
    worst = max(rows, key=lambda r: r["abs_err"])
    passed = worst["abs_err"] < 1e-12
    return dict(gate="G2", passed=passed, threshold=1e-12, rows=rows, worst=worst)


def _bessel_table(orders, u, cache=None):
    """J_k, J_k', J_k'' for every k in `orders`, vectorised over u."""
    orders = sorted(set(int(k) for k in orders))
    u_arr = np.asarray(u, dtype=float)
    key = None
    if cache is not None and u_arr.size == 1:
        key = ("d2", float(u_arr.ravel()[0]), orders[0], orders[-1],
               len(orders))
        hit = cache.get(key)
        if hit is not None:
            return hit
    lo, hi = orders[0] - 2, orders[-1] + 2
    ks = np.arange(lo, hi + 1)
    if u_arr.ndim == 0:
        rows = jv(ks, u_arr)
    else:
        rows = jv(ks[:, None], u_arr[None, :])
    raw = {int(k): rows[i] for i, k in enumerate(ks)}
    J, dJ, ddJ = {}, {}, {}
    for k in orders:
        J[k] = raw[k]
        dJ[k] = 0.5 * (raw[k - 1] - raw[k + 1])
        dJ_km1 = 0.5 * (raw[k - 2] - raw[k])
        dJ_kp1 = 0.5 * (raw[k] - raw[k + 2])
        ddJ[k] = 0.5 * (dJ_km1 - dJ_kp1)
    if key is not None:
        _memo_store(cache, key, (J, dJ, ddJ))
    return J, dJ, ddJ


def _bessel_values_cached(solver, hi, u):
    """Raw J_k(u) for k = 0..hi as one array, memoised on the solver for scalar u
    (the root scan's case)."""
    key = ("val", float(u), hi)
    hit = solver._bessel_cache.get(key)
    if hit is None:
        hit = jv(np.arange(hi + 1), u)
        _memo_store(solver._bessel_cache, key, hit)
    return hit


def radial_value(solver: MathieuAngularTridiagonal, n: int, parity: str,
                 mu: float, q: float) -> float:
    """Value-only Mc/Ms at scalar `mu` -- the root scan's boundary function."""
    cname, j = _locate(n, parity)
    coeffs = solver.coeffs_natural(cname, j)
    K = len(coeffs)
    mu_a = np.atleast_1d(np.asarray(mu, dtype=float))
    u1 = np.sqrt(q) * np.exp(-mu_a)
    u2 = np.sqrt(q) * np.exp(mu_a)
    u1f, u2f = float(u1[0]), float(u2[0])

    if cname == "ce_even":
        J1 = _bessel_values_cached(solver, K - 1, u1f)
        J2 = _bessel_values_cached(solver, K - 1, u2f)
        val = 0.0
        for k in range(K):
            sign = (-1.0) ** k
            val = val + sign * coeffs[k] * (J1[k] * J2[k])
    elif cname == "ce_odd":
        J1 = _bessel_values_cached(solver, K, u1f)
        J2 = _bessel_values_cached(solver, K, u2f)
        val = 0.0
        for k in range(K):
            sign = (-1.0) ** k
            v1 = J1[k] * J2[k + 1]
            v2 = J1[k + 1] * J2[k]
            val = val + sign * coeffs[k] * (v1 + v2)
    elif cname == "se_odd":
        J1 = _bessel_values_cached(solver, K, u1f)
        J2 = _bessel_values_cached(solver, K, u2f)
        val = 0.0
        for k in range(K):
            sign = (-1.0) ** k
            v1 = J1[k] * J2[k + 1]
            v2 = J1[k + 1] * J2[k]
            val = val + sign * coeffs[k] * (v1 - v2)
    elif cname == "se_even":
        J1 = _bessel_values_cached(solver, K + 1, u1f)
        J2 = _bessel_values_cached(solver, K + 1, u2f)
        val = 0.0
        for k in range(K):
            sign = (-1.0) ** k
            v1 = J1[k] * J2[k + 2]
            v2 = J1[k + 2] * J2[k]
            val = val + sign * coeffs[k] * (v1 - v2)
    else:
        raise ValueError(f"unknown class {cname!r}")

    p = ((-1.0) ** j) / coeffs[0]
    return float(p * val)


def _cross_term_deriv2(Ja, dJa, ddJa, Jb, dJb, ddJb, u1, u2):
    """Value and first two mu-derivatives of J_a(u1(mu)) * J_b(u2(mu))."""
    val = Ja * Jb
    dval = -u1 * dJa * Jb + u2 * Ja * dJb
    ddval = (u1 ** 2) * ddJa * Jb + u1 * dJa * Jb \
        - 2.0 * u1 * u2 * dJa * dJb \
        + (u2 ** 2) * Ja * ddJb + u2 * Ja * dJb
    return val, dval, ddval


def radial_value_deriv2(solver: MathieuAngularTridiagonal, n: int, parity: str,
                        mu, q: float) -> tuple:
    """Radial Mathieu function and its first two mu-derivatives from the
    Bessel-product series."""
    cname, j = _locate(n, parity)
    coeffs = solver.coeffs_natural(cname, j)
    K = len(coeffs)
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    u1 = np.sqrt(q) * np.exp(-mu)
    u2 = np.sqrt(q) * np.exp(mu)

    if cname == "ce_even":         # Mc_{2m}: sum_k (-1)^k A_2k Jk(u1)Jk(u2)
        orders = list(range(K))
        J1, dJ1, ddJ1 = _bessel_table(orders, u1, cache=solver._bessel_cache)
        J2, dJ2, ddJ2 = _bessel_table(orders, u2, cache=solver._bessel_cache)
        val = dval = ddval = 0.0
        for k in orders:
            sign = (-1.0) ** k
            v, dv, ddv = _cross_term_deriv2(J1[k], dJ1[k], ddJ1[k],
                                            J2[k], dJ2[k], ddJ2[k], u1, u2)
            val = val + sign * coeffs[k] * v
            dval = dval + sign * coeffs[k] * dv
            ddval = ddval + sign * coeffs[k] * ddv
        p = ((-1.0) ** j) / coeffs[0]

    elif cname == "ce_odd":
        orders = list(range(K + 1))
        J1, dJ1, ddJ1 = _bessel_table(orders, u1, cache=solver._bessel_cache)
        J2, dJ2, ddJ2 = _bessel_table(orders, u2, cache=solver._bessel_cache)
        val = dval = ddval = 0.0
        for k in range(K):
            sign = (-1.0) ** k
            v1, dv1, ddv1 = _cross_term_deriv2(J1[k], dJ1[k], ddJ1[k],
                                               J2[k + 1], dJ2[k + 1], ddJ2[k + 1], u1, u2)
            v2, dv2, ddv2 = _cross_term_deriv2(J1[k + 1], dJ1[k + 1], ddJ1[k + 1],
                                               J2[k], dJ2[k], ddJ2[k], u1, u2)
            val = val + sign * coeffs[k] * (v1 + v2)
            dval = dval + sign * coeffs[k] * (dv1 + dv2)
            ddval = ddval + sign * coeffs[k] * (ddv1 + ddv2)
        p = ((-1.0) ** j) / coeffs[0]

    elif cname == "se_odd":
        orders = list(range(K + 1))
        J1, dJ1, ddJ1 = _bessel_table(orders, u1, cache=solver._bessel_cache)
        J2, dJ2, ddJ2 = _bessel_table(orders, u2, cache=solver._bessel_cache)
        val = dval = ddval = 0.0
        for k in range(K):
            sign = (-1.0) ** k
            v1, dv1, ddv1 = _cross_term_deriv2(J1[k], dJ1[k], ddJ1[k],
                                               J2[k + 1], dJ2[k + 1], ddJ2[k + 1], u1, u2)
            v2, dv2, ddv2 = _cross_term_deriv2(J1[k + 1], dJ1[k + 1], ddJ1[k + 1],
                                               J2[k], dJ2[k], ddJ2[k], u1, u2)
            val = val + sign * coeffs[k] * (v1 - v2)
            dval = dval + sign * coeffs[k] * (dv1 - dv2)
            ddval = ddval + sign * coeffs[k] * (ddv1 - ddv2)
        p = ((-1.0) ** j) / coeffs[0]

    elif cname == "se_even":
        orders = list(range(K + 2))
        J1, dJ1, ddJ1 = _bessel_table(orders, u1, cache=solver._bessel_cache)
        J2, dJ2, ddJ2 = _bessel_table(orders, u2, cache=solver._bessel_cache)
        val = dval = ddval = 0.0
        for k in range(K):
            sign = (-1.0) ** k
            v1, dv1, ddv1 = _cross_term_deriv2(J1[k], dJ1[k], ddJ1[k],
                                               J2[k + 2], dJ2[k + 2], ddJ2[k + 2], u1, u2)
            v2, dv2, ddv2 = _cross_term_deriv2(J1[k + 2], dJ1[k + 2], ddJ1[k + 2],
                                               J2[k], dJ2[k], ddJ2[k], u1, u2)
            val = val + sign * coeffs[k] * (v1 - v2)
            dval = dval + sign * coeffs[k] * (dv1 - dv2)
            ddval = ddval + sign * coeffs[k] * (ddv1 - ddv2)
        p = ((-1.0) ** j) / coeffs[0]
    else:
        raise ValueError(f"unknown class {cname!r}")

    val, dval, ddval = p * val, p * dval, p * ddval
    scalar_in = np.isscalar(mu) or np.asarray(mu).ndim == 0
    if scalar_in or (hasattr(val, "shape") and val.shape == (1,)):
        return float(np.ravel(val)[0]), float(np.ravel(dval)[0]), float(np.ravel(ddval)[0])
    return val, dval, ddval


def gate_G3_radial_ode_residual(q: float, n_max: int, mu0: float, *,
                                n_mu: int = 60) -> dict:
    """||y'' - (a - 2q cosh2mu) y|| / scale < 1e-10 on mu in [0, mu0 + margin], per
    class, across q."""
    solver = MathieuAngularTridiagonal(q=q, n_max=n_max)
    mu = np.linspace(1e-3, mu0 * 1.1 + 0.05, n_mu)
    abs_floor = 1e-10
    rows = []
    for n in range(n_max + 1):
        for parity in ("even", "odd"):
            if parity == "odd" and n == 0:
                continue
            cname, j = _locate(n, parity)
            a = solver.a_value(cname, j)
            val, _, ddval = radial_value_deriv2(solver, n, parity, mu, q)
            term2 = (a - 2 * q * np.cosh(2 * mu)) * val
            residual = ddval - term2
            scale = np.maximum(np.abs(ddval), np.abs(term2))
            scale = np.maximum(scale, 1e-300)
            rel = float(np.max(np.abs(residual) / scale))
            abs_max = float(np.max(np.abs(residual)))
            gate_val = min(rel, abs_max)
            rows.append(dict(n=n, parity=parity, q=q, residual_rel=rel,
                             residual_abs_max=abs_max, gate_val=gate_val))
    worst = max(rows, key=lambda r: r["gate_val"])
    passed = worst["gate_val"] < abs_floor
    return dict(gate="G3", passed=passed, threshold=1e-10, rows=rows, worst=worst)


def gate_G4_low_q_scipy_crosscheck(*, q_values=(0.5, 2.0, 5.0), n_max: int = 8,
                                   mu0: float = 0.6, nu0: float = 0.7) -> dict:
    """low-q cross-check vs SciPy (its trusted regime): values AND first derivatives
    of ce/se (angular) and Mc/Ms (radial) agree to <= 1e-8 absolute, after the
    (-1)^j sign fix (documented on `radial_value_deriv2`, determined empirically
    by this exact check, not guessed in advance)."""
    from scipy.special import mathieu_cem, mathieu_sem, mathieu_modcem1, mathieu_modsem1

    rows = []
    for q in q_values:
        solver = MathieuAngularTridiagonal(q=q, n_max=n_max)
        for n in range(n_max + 1):
            for parity in ("even", "odd"):
                if parity == "odd" and n == 0:
                    continue
                cname, j = _locate(n, parity)
                fn_ang = mathieu_cem if parity == "even" else mathieu_sem
                val_a, dval_a, _ = angular_value_deriv2(solver, n, parity, nu0)
                sp_val_a = fn_ang(n, q, np.rad2deg(nu0))[0]
                h = 1e-6
                sp_dval_a = (fn_ang(n, q, np.rad2deg(nu0 + h))[0]
                            - fn_ang(n, q, np.rad2deg(nu0 - h))[0]) / (2 * h)
                err_val_a = abs(val_a - sp_val_a)
                err_dval_a = abs(dval_a - sp_dval_a)

                fn_rad = mathieu_modcem1 if parity == "even" else mathieu_modsem1
                val_r, dval_r, _ = radial_value_deriv2(solver, n, parity, mu0, q)
                sp_val_r, sp_dval_r = fn_rad(n, q, mu0)
                err_val_r = abs(val_r - sp_val_r)
                err_dval_r = abs(dval_r - sp_dval_r)

                rows.append(dict(q=q, n=n, parity=parity,
                                 err_angular_value=err_val_a,
                                 err_angular_deriv=err_dval_a,
                                 err_radial_value=err_val_r,
                                 err_radial_deriv=err_dval_r))
    worst = max(rows, key=lambda r: max(r["err_angular_value"], r["err_angular_deriv"],
                                        r["err_radial_value"], r["err_radial_deriv"]))
    worst_val = max(worst["err_angular_value"], worst["err_angular_deriv"],
                    worst["err_radial_value"], worst["err_radial_deriv"])
    passed = worst_val < 1e-8
    return dict(gate="G4", passed=passed, threshold=1e-8, rows=rows,
               worst=worst, worst_val=worst_val)
