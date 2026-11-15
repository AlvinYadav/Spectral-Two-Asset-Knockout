"""Mathieu eigenbasis for the Euclidean-circle domain."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import brentq

from legacy.mathieu import (
    Ce, Se, ce, se, Ce_deriv, Se_deriv, ce_deriv, se_deriv,
)
from scipy.special import jv

from ..mathieu_backend import (
    MathieuAngularTridiagonal,
    _locate,
    angular_value_deriv2,
    radial_value,
    radial_value_deriv2,
    radial_value_deriv2_vec,
)

from .base import EigenBasis, EigenMode, gram_dedupe


# Brent polishing tolerance for every scanned root. Part of the root-cache
# key, so a change here can never be served roots polished at the old value.
_ROOT_XTOL = 1e-10

_ROOT_CACHE = {}
_ROOT_CACHE_MAX = 64


def clear_root_cache():
    """Empty the process-wide root memo (cold-fit benchmarks and tests)."""
    _ROOT_CACHE.clear()


@dataclass
class MathieuBasis(EigenBasis):
    """Mathieu basis on the elliptic domain induced by a Euclidean circle."""
    N_trunc: int = 50
    N_angular_max: int = 6
    M_radial_max: int = 4
    q_min: float = 0.1
    q_max: float = 100.0
    q_scan_points: int = 1500
    N_mu_quad: int = 40
    N_nu_quad: int = 96
    backend: str = "scipy"
    tensor_design: bool = True
    dedup_threshold: float = 0.999
    root_cache: bool = True
    # Truncation rule.
    #   "index"      angular orders n <= N_angular_max, the first
    #                M_radial_max roots of each order and parity, scan on
    #                [q_min, q_max] with q_scan_points: the earlier rule, kept
    #                to reproduce Table 6.5 and the sealed records. Default, so
    #                every config without the key behaves as before.
    #   "eigenvalue" every mode with lambda^2 < Lambda, where
    #                Lambda = lambda_1^2 + (2 / tau_min) ln(1 / eps_trunc),
    #                tau_min = tau_min_frac * T (or lambda2_max if given).
    #                The scan runs to q_max = c^2 Lambda / 4 at
    #                q_scan_density points per unit q; N_angular_max,
    #                M_radial_max, q_max and q_scan_points are not used.
    #                Production configs select it explicitly. Tridiagonal
    #                backend only.
    truncation: str = "index"
    tau_min_frac: float = 0.25
    eps_trunc: float = 1e-6
    lambda2_max: float = None
    q_scan_density: float = 15.0

    # Filled by .fit():
    modes: list = field(default_factory=list)
    market: object = None
    domain: object = None
    c_focal: float = 0.0
    mu_0: float = 0.0
    A_semi: float = 0.0
    B_semi: float = 0.0
    gram_matrix: np.ndarray = field(default=None, repr=False)
    mode_ledger: dict = field(default=None, repr=False)
    # Eigenvalue-mode record: lambda2_first, lambda2_cut, q_scan_max,
    # n_max (largest scanned order), families, grid points evaluated and
    # sign changes rejected by the turning-point test. None in index mode.
    truncation_info: dict = field(default=None, repr=False)
    _last_dedup_dropped: list = field(default_factory=list, repr=False)
    _backend_solver_cache: dict = field(default_factory=dict, repr=False)
    _sep_eval_cache: dict = field(default_factory=dict, repr=False)
    # True when the last fit took its roots from the memo (logging only).
    _root_cache_hit: bool = field(default=False, repr=False)
    # Order the angular tridiagonal truncation K is sized for: N_angular_max
    # in index mode, the largest scanned order in eigenvalue mode.
    _n_max_solver: int = field(default=0, repr=False)

    # ── Setup ────────────────────────────────────────────────────
    def fit(self, market, domain):
        """Find eigenvalues, compute L² norms."""
        if self.backend not in {"scipy", "tridiagonal"}:
            raise ValueError(
                f"unknown Mathieu backend {self.backend!r}; expected "
                "'scipy' or 'tridiagonal'"
            )
        if self.truncation not in {"index", "eigenvalue"}:
            raise ValueError(
                f"unknown truncation {self.truncation!r}; expected 'index' or "
                "'eigenvalue'"
            )
        if self.truncation == "eigenvalue" and self.backend != "tridiagonal":
            raise ValueError(
                "truncation='eigenvalue' requires backend='tridiagonal'"
            )
        self._backend_solver_cache.clear()
        self._sep_eval_cache.clear()
        self._n_max_solver = int(self.N_angular_max)
        self.truncation_info = None
        self.market = market
        self.domain = domain
        g = domain.geometry(market)
        self.c_focal = g["c_focal"]
        self.mu_0 = g["mu_0"]
        self.A_semi = g["A_semi"]
        self.B_semi = g["B_semi"]

        ratio = min(self.A_semi, self.B_semi) / max(self.A_semi, self.B_semi)
        ecc_sq = 1.0 - ratio ** 2
        if ecc_sq < 1e-3:
            raise ValueError(
                f"whitened ellipse is near-isotropic (e^2 = {ecc_sq:.2e} < "
                "1e-3): elliptic coordinates degenerate (mu_0 -> inf, q -> 0);"
                " use the circular/Bessel (MahalanobisEllipse) solver for this"
                " geometry"
            )

        self._find_eigenvalues()
        constructed = len(self.modes)
        self._compute_norms()
        self._dedupe_near_degenerate_modes()
        self._record_mode_ledger(constructed)
        return self

    # ── Backend-neutral special-function evaluation ─────────────
    def _angular_solver(self, q: float, n: int = None) -> MathieuAngularTridiagonal:
        """Return the cached tridiagonal angular solve for one root q.

        Index mode sizes every solve for N_angular_max, as before. Eigenvalue
        mode sizes it for the order n being evaluated: the truncation starts
        at K = max(2n + 16, 32) and the backend's doubling rule checks the
        coefficient tail of every column up to that order, so a low-order
        mode is not evaluated with the K of the highest retained order."""
        if self.truncation == "eigenvalue" and n is not None:
            order = int(n)
            key = (float(q), order)
        else:
            order = self._n_max_solver
            key = float(q)
        solver = self._backend_solver_cache.get(key)
        if solver is None:
            solver = MathieuAngularTridiagonal(q=float(q), n_max=order)
            self._backend_solver_cache[key] = solver
        return solver

    def _separated_value_deriv2(self, n, q, parity, mu, nu):
        """Return radial/angular values and first two native derivatives."""
        if self.backend == "tridiagonal":
            key = None
            if np.ndim(mu) == 0 and np.ndim(nu) == 0:
                key = (int(n), float(q), parity, float(mu), float(nu))
                hit = self._sep_eval_cache.get(key)
                if hit is not None:
                    return hit
            solver = self._angular_solver(q, n)
            if self.truncation == "eigenvalue":
                Rv, Rd, Rdd = radial_value_deriv2_vec(solver, n, parity, mu, q)
            else:
                Rv, Rd, Rdd = radial_value_deriv2(solver, n, parity, mu, q)
            Av, Ad, Add = angular_value_deriv2(solver, n, parity, nu)
            out = (Rv, Rd, Rdd, Av, Ad, Add)
            if key is not None:
                if len(self._sep_eval_cache) >= 8192:
                    self._sep_eval_cache.clear()
                self._sep_eval_cache[key] = out
            return out

        NT = self.N_trunc
        mu_arr = np.asarray(mu)
        nu_arr = np.asarray(nu)
        if parity == "even":
            Rv = np.array([Ce(n, q, x, NT) for x in np.atleast_1d(mu_arr)])
            Rd = np.array([Ce_deriv(n, q, x, NT) for x in np.atleast_1d(mu_arr)])
            Av = np.array([ce(n, q, x, NT) for x in np.atleast_1d(nu_arr)])
            Ad = np.array([ce_deriv(n, q, x, NT) for x in np.atleast_1d(nu_arr)])
        else:
            Rv = np.array([Se(n, q, x, NT) for x in np.atleast_1d(mu_arr)])
            Rd = np.array([Se_deriv(n, q, x, NT) for x in np.atleast_1d(mu_arr)])
            Av = np.array([se(n, q, x, NT) for x in np.atleast_1d(nu_arr)])
            Ad = np.array([se_deriv(n, q, x, NT) for x in np.atleast_1d(nu_arr)])
        from scipy.special import mathieu_a, mathieu_b
        a = mathieu_a(n, q) if parity == "even" else mathieu_b(n, q)
        Rdd = (a - 2 * q * np.cosh(2 * np.atleast_1d(mu_arr))) * Rv
        Add = -(a - 2 * q * np.cos(2 * np.atleast_1d(nu_arr))) * Av
        scalar_mu = mu_arr.ndim == 0
        scalar_nu = nu_arr.ndim == 0
        if scalar_mu:
            Rv, Rd, Rdd = float(Rv[0]), float(Rd[0]), float(Rdd[0])
        if scalar_nu:
            Av, Ad, Add = float(Av[0]), float(Ad[0]), float(Add[0])
        return Rv, Rd, Rdd, Av, Ad, Add

    def _boundary_value(self, n: int, q: float, parity: str) -> float:
        if self.backend == "tridiagonal":
            return radial_value(self._angular_solver(q, n), n, parity,
                                self.mu_0, q)
        Rv, _, _, _, _, _ = self._separated_value_deriv2(
            n, q, parity, self.mu_0, 0.0
        )
        return float(Rv)

    def _record_mode_ledger(self, constructed: int):
        """Record how many modes were found, kept and used (diagnostics only)."""
        low_norm = [md.meta for md in self.modes if md.norm <= 1e-6]
        self.mode_ledger = dict(
            basis="MathieuBasis",
            candidates=constructed,
            constructed=constructed,
            dedup_dropped=self._last_dedup_dropped,
            n_dedup_dropped=len(self._last_dedup_dropped),
            post_dedup=len(self.modes),
            downstream_low_norm=low_norm,
            n_downstream_low_norm=len(low_norm),
            active=int(sum(1 for md in self.modes if md.norm > 1e-6)),
        )

    def _dedupe_near_degenerate_modes(self):
        """Check that the modes are orthonormal on the quadrature grid and drop
        near-duplicates."""
        mu_nodes, mu_weights = self._mu_quad
        nu_nodes = self._nu_quad
        dnu = self._dnu
        sinh2 = np.sinh(mu_nodes) ** 2
        sin2 = np.sin(nu_nodes) ** 2
        h2 = self.c_focal ** 2 * (sinh2[:, None] + sin2[None, :])
        quad_weight_flat = (mu_weights[:, None] * dnu * h2).ravel()

        psi_flat = np.empty((len(self.modes), quad_weight_flat.shape[0]))
        for k, md in enumerate(self.modes):
            n, q, parity = md.meta["n"], md.meta["q"], md.meta["parity"]
            R_vals, _, _, A_vals, _, _ = self._separated_value_deriv2(
                n, q, parity, mu_nodes, nu_nodes
            )
            norm = md.norm if md.norm > 1e-15 else 1.0
            psi_flat[k] = (R_vals[:, None] * A_vals[None, :]).ravel() / norm

        self.modes, self.gram_matrix, self._last_dedup_dropped = gram_dedupe(
            self.modes, psi_flat, quad_weight_flat, label="MathieuBasis",
            dup_threshold=self.dedup_threshold,
        )

    # ── Eigenvalue search ────────────────────────────────────────
    def _families(self):
        """(n, parity) for every scanned family, in scan order."""
        families = [(n, "even") for n in range(self.N_angular_max + 1)]
        families += [(n, "odd") for n in range(1, self.N_angular_max + 1)]
        return families

    def _root_cache_key(self, families):
        """Everything the scanned roots depend on. c_focal is deliberately absent: it
        only rescales k_sq, which each fit recomputes."""
        return (self.backend, float(self.mu_0), float(self.q_min),
                float(self.q_max), int(self.q_scan_points), int(self.N_trunc),
                int(self.N_angular_max), int(self.M_radial_max), _ROOT_XTOL,
                tuple(families))

    def _find_eigenvalues(self):
        if self.truncation == "eigenvalue":
            self._find_eigenvalues_cutoff()
            return
        families = self._families()
        key = self._root_cache_key(families) if self.root_cache else None
        roots = _ROOT_CACHE.get(key) if key is not None else None
        self._root_cache_hit = roots is not None
        if roots is None:
            if self.backend == "scipy":
                roots = self._scan_roots_scipy(families)
            else:
                roots = self._scan_roots_tridiagonal(families)
            if key is not None:
                if len(_ROOT_CACHE) >= _ROOT_CACHE_MAX:
                    _ROOT_CACHE.clear()
                _ROOT_CACHE[key] = roots
        self.modes = self._modes_from_roots(roots)

    def _modes_from_roots(self, roots):
        """Fresh EigenModes for this fit's c_focal, sorted by k_sq."""
        c2 = self.c_focal ** 2
        modes = [
            EigenMode(idx=(n, m, parity), k_sq=4 * qr / c2,
                      meta=dict(n=n, m=m, parity=parity, q=qr))
            for n, m, parity, qr in roots
        ]
        modes.sort(key=lambda m: m.k_sq)
        return modes

    def _append_roots(self, roots, n, parity, q_grid, vals):
        """Append the first requested true sign-change roots for one order as (n, m,
        parity, q)."""
        found = 0
        for i in range(len(vals) - 1):
            if found >= self.M_radial_max:
                break
            if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
                continue
            if vals[i] * vals[i + 1] < 0:
                try:
                    qr = brentq(
                        lambda qi: self._boundary_value(n, qi, parity),
                        q_grid[i], q_grid[i + 1], xtol=_ROOT_XTOL,
                    )
                except (ValueError, RuntimeError, FloatingPointError):
                    continue
                found += 1
                roots.append((n, found, parity, qr))

    def _scan_roots_scipy(self, families):
        roots = []
        q_grid = np.linspace(0.01, self.q_max, self.q_scan_points)
        NT = self.N_trunc
        mu0 = self.mu_0

        for n, parity in families:
            radial = Ce if parity == "even" else Se
            vals = np.array([radial(n, qi, mu0, NT) for qi in q_grid])
            self._append_roots(roots, n, parity, q_grid, vals)
        return tuple(roots)

    def _scan_roots_tridiagonal(self, families):
        """Find the eigenvalues by scanning the radial boundary function over q."""
        q_grid = np.linspace(self.q_min, self.q_max, self.q_scan_points)
        samples = {key: np.empty(len(q_grid)) for key in families}

        for i, q in enumerate(q_grid):
            solver = MathieuAngularTridiagonal(q=q, n_max=self.N_angular_max)
            for n, parity in families:
                samples[(n, parity)][i] = radial_value(
                    solver, n, parity, self.mu_0, q
                )

        roots = []
        for n, parity in families:
            self._append_roots(roots, n, parity, q_grid, samples[(n, parity)])
        return tuple(roots)

    # ── Eigenvalue cutoff ────────────────────────────────────────
    # Each mode decays like exp(-lambda^2 tau / 2), so the truncation error
    # is governed by the smallest omitted eigenvalue, and the retained set is
    # every mode with lambda^2 < Lambda. Two facts about the radial equation
    # M'' = (a - 2q cosh 2mu) M, M'(0) = 0 (even) or M(0) = 0 (odd), are
    # used:
    #   * turning point: if char_n(q) >= 2 q cosh 2mu_0, M cannot vanish at
    #     mu_0, so a root of family (n, parity) needs char_n(q) < 2q cosh 2mu_0;
    #   * monotonicity: char_n(q) - 2q cosh 2mu_0 decreases strictly in q and
    #     increases in n, so once char_n(q_max) >= 2 q_max cosh 2mu_0 for both
    #     parities, order n and every higher order have no root below q_max,
    #     and each family needs scanning only above its own crossing q*_n.
    @staticmethod
    def _char(solver, n, parity):
        """Characteristic value a_n (even) or b_n (odd) at the solver's q."""
        cname, j = _locate(n, parity)
        return solver.a_value(cname, j)

    def _radial0(self, q):
        """Ce_0(mu_0, q) from a minimal solve, for the lambda_1 search."""
        return radial_value(MathieuAngularTridiagonal(q=q, n_max=0), 0,
                            "even", self.mu_0, q)

    def _first_root_order0(self):
        """q of the lowest Dirichlet mode: the first root of the (0, even)
        family, found by scanning upward from q_min at the cutoff spacing and
        refining by Brent. Its radial function must have no zero inside
        (0, mu_0); otherwise a root below q_min was missed."""
        h = 1.0 / self.q_scan_density
        q_prev = float(self.q_min)
        f_prev = self._radial0(q_prev)
        k = 1
        while True:
            q = float(self.q_min) + k * h
            f = self._radial0(q)
            if np.isfinite(f_prev) and np.isfinite(f) and f_prev * f < 0:
                break
            q_prev, f_prev = q, f
            k += 1
            if k > 10 ** 6:
                raise RuntimeError("no root of the order-0 even family found")
        q1 = brentq(self._radial0, q_prev, q, xtol=_ROOT_XTOL)
        mu = np.linspace(0.0, self.mu_0, 202)[1:-1]
        Rv = radial_value_deriv2(MathieuAngularTridiagonal(q=q1, n_max=0), 0,
                                 "even", mu, q1)[0]
        if np.any(np.sign(Rv[:-1]) * np.sign(Rv[1:]) < 0):
            raise RuntimeError(
                f"the first order-0 root found (q = {q1:.6g}) has an interior "
                f"zero: a root below q_min = {self.q_min} was missed"
            )
        return float(q1)

    def _stopping_order(self, q_hi, two_cosh):
        """Smallest n >= 1 with a_n(q_hi) and b_n(q_hi) both at or above
        2 q_hi cosh 2mu_0: no family of that order or above has a root."""
        thr = two_cosh * q_hi
        n_try = 16
        while True:
            solver = MathieuAngularTridiagonal(q=q_hi, n_max=n_try)
            for n in range(1, n_try + 1):
                if (self._char(solver, n, "even") >= thr
                        and self._char(solver, n, "odd") >= thr):
                    return n
            n_try *= 2
            if n_try > 1024:
                raise RuntimeError("stopping order not found below n = 1024")

    def _find_eigenvalues_cutoff(self):
        """Roots of every family below q_max = c^2 Lambda / 4.

        The scan grid is fixed, q_k = q_min + k h with h = 1 / q_scan_density,
        whatever the limit, so a scan to a smaller limit is a prefix of a scan
        to a larger one. The memo is keyed on the ellipse's shape (mu_0) and the
        scan settings only and holds the roots up to the largest limit scanned:
        a solve with a smaller limit (a smaller radius, a longer maturity) takes
        its roots from it by filtering, and a larger limit extends the stored
        scan from where it stopped. Brent refines each bracket with the
        order-sized solve of `_angular_solver`, so a bracket refined once gives
        the same root it would give in a fresh scan; only the sign tests at the
        grid points use a shared solve sized for the largest scanned order.
        """
        c2 = self.c_focal ** 2
        two_cosh = 2.0 * np.cosh(2.0 * self.mu_0)
        h = 1.0 / self.q_scan_density
        key = None
        entry = None
        if self.root_cache:
            key = ("eigenvalue", self.backend, float(self.mu_0), float(self.q_min),
                   float(self.q_scan_density), _ROOT_XTOL)
            entry = _ROOT_CACHE.get(key)
        if entry is None:
            entry = dict(q1=self._first_root_order0(), k_last=-1, families={},
                         roots=[], evaluations=0, sign_changes=0,
                         rejected_turning_point=0, brent_failures=0, n_max=0)
            fresh = True
        else:
            fresh = False
        lam1 = 4.0 * entry["q1"] / c2
        if self.lambda2_max is not None:
            lam_cut, tau_min = float(self.lambda2_max), None
        else:
            tau_min = self.tau_min_frac * self.market.T
            lam_cut = lam1 + (2.0 / tau_min) * np.log(1.0 / self.eps_trunc)
        if lam_cut <= lam1:
            raise ValueError(
                f"cutoff Lambda = {lam_cut:.6g} is not above lambda_1^2 = "
                f"{lam1:.6g}"
            )
        q_hi = c2 * lam_cut / 4.0
        k_need = int(np.ceil((q_hi - self.q_min) / h))
        extended = entry["k_last"] < k_need
        if extended:
            self._scan_cutoff_extend(entry, k_need, two_cosh, h)
        if key is not None:
            if key not in _ROOT_CACHE and len(_ROOT_CACHE) >= _ROOT_CACHE_MAX:
                _ROOT_CACHE.clear()
            _ROOT_CACHE[key] = entry
        self._root_cache_hit = not fresh and not extended
        self._n_max_solver = entry["n_max"]
        roots = tuple(r for r in entry["roots"] if r[3] < q_hi)
        self.modes = self._modes_from_roots(roots)
        self.truncation_info = dict(
            rule="eigenvalue", tau_min=tau_min, eps=float(self.eps_trunc),
            lambda2_first=float(lam1), lambda2_cut=float(lam_cut),
            q_scan_max=float(q_hi), q_scan_points=k_need + 1,
            n_max=entry["n_max"], n_roots=len(roots),
            memo=("scanned" if fresh else ("extended" if extended else "hit")),
            evaluations=entry["evaluations"], sign_changes=entry["sign_changes"],
            rejected_turning_point=entry["rejected_turning_point"],
            brent_failures=entry["brent_failures"],
        )

    def _scan_cutoff_extend(self, entry, k_need, two_cosh, h):
        """Continue the scan of `entry` from grid index k_last + 1 to k_need.
        Each family is evaluated only where the turning-point condition allows a
        root, plus the grid point below, so a root just above its crossing is
        still bracketed; every refined root is checked against the condition
        again and rejected if it fails. Updates `entry` in place."""
        q_of = lambda k: float(self.q_min) + k * h
        n_stop = self._stopping_order(q_of(k_need), two_cosh)
        n_max = max(n_stop - 1, entry["n_max"])
        entry["n_max"] = n_max
        families = [(n, "even") for n in range(n_max + 1)]
        families += [(n, "odd") for n in range(1, n_max + 1)]
        fam = entry["families"]
        for f in families:
            fam.setdefault(f, dict(started=False, m=0))
        loc = {f: _locate(*f) for f in families}
        k0 = entry["k_last"] + 1

        def values(solver, q, fs):
            want = {}
            for f in fs:
                want.setdefault(loc[f][0], []).append(loc[f][1])
            got = self._radial_values_all(solver, self.mu_0, q, want)
            entry["evaluations"] += len(fs)
            return {f: got[loc[f]] for f in fs}

        prev = {}
        prev_solver = None
        if k0 > 0:      # resume: the left end of the first new bracket
            prev_solver = MathieuAngularTridiagonal(q=q_of(k0 - 1), n_max=n_max)
            started = [f for f in families if fam[f]["started"]]
            prev = values(prev_solver, q_of(k0 - 1), started)
        for k in range(k0, k_need + 1):
            q = q_of(k)
            solver = MathieuAngularTridiagonal(q=q, n_max=n_max)
            thr = two_cosh * q
            active, newly = [], []
            for f in families:
                if not fam[f]["started"]:
                    if self._char(solver, *f) >= thr:
                        continue
                    fam[f]["started"] = True
                    if prev_solver is not None:
                        newly.append(f)
                active.append(f)
            if newly:
                prev.update(values(prev_solver, q_of(k - 1), newly))
            cur = values(solver, q, active)
            for f in active:
                a, b = prev.get(f), cur[f]
                if a is None or not (np.isfinite(a) and np.isfinite(b)) or a * b >= 0:
                    continue
                n, parity = f
                entry["sign_changes"] += 1
                try:
                    qr = brentq(lambda qi: self._boundary_value(n, qi, parity),
                                q_of(k - 1), q, xtol=_ROOT_XTOL)
                except (ValueError, RuntimeError, FloatingPointError):
                    entry["brent_failures"] += 1
                    continue
                if self._char(self._angular_solver(qr, n), n, parity) >= two_cosh * qr:
                    entry["rejected_turning_point"] += 1
                    continue
                fam[f]["m"] += 1
                entry["roots"].append((n, fam[f]["m"], parity, qr))
            prev = cur
            prev_solver = solver
        entry["k_last"] = k_need

    @staticmethod
    def _radial_values_all(solver, mu0, q, wanted):
        """Radial boundary values of several orders at once, for the cutoff
        scan: the same Bessel-product series as `radial_value`, with the
        same coefficients and normalisation p = (-1)^j / c_0, summed as one
        matrix product per symmetry class instead of a Python loop per
        order. `wanted` maps class name -> eigenvalue indices j. Returns
        {(class, j): value}. Values agree with `radial_value` to rounding;
        Brent refinement still uses `radial_value` itself."""
        K = solver.K
        u1 = np.sqrt(q) * np.exp(-mu0)
        u2 = np.sqrt(q) * np.exp(mu0)
        J1 = jv(np.arange(K + 2), u1)
        J2 = jv(np.arange(K + 2), u2)
        k = np.arange(K)
        sign = (-1.0) ** k
        out = {}
        for cname, js in wanted.items():
            if not js:
                continue
            if cname == "ce_even":
                T = J1[k] * J2[k]
            elif cname == "ce_odd":
                T = J1[k] * J2[k + 1] + J1[k + 1] * J2[k]
            elif cname == "se_odd":
                T = J1[k] * J2[k + 1] - J1[k + 1] * J2[k]
            else:
                T = J1[k] * J2[k + 2] - J1[k + 2] * J2[k]
            C = solver.coeffs[cname][:, js].copy()
            if cname == "ce_even":
                C[0, :] /= np.sqrt(2.0)
            vals = (sign * T) @ C
            p = ((-1.0) ** np.asarray(js)) / C[0, :]
            for j, v in zip(js, p * vals):
                out[(cname, j)] = float(v)
        return out

    def radial_zero_counts(self, n_points: int = 2000):
        """Sign changes of each mode's radial function at n_points interior
        points of (0, mu_0). By Sturm's theorem the m-th root of a family has
        exactly m - 1 (zero-count gate)."""
        mu = np.linspace(0.0, self.mu_0, n_points + 2)[1:-1]
        rows = []
        for md in self.modes:
            n, q, parity, m = (md.meta["n"], md.meta["q"], md.meta["parity"],
                               md.meta["m"])
            Rv = self._separated_value_deriv2(n, q, parity, mu, 0.0)[0]
            zeros = int(np.sum(np.sign(Rv[:-1]) * np.sign(Rv[1:]) < 0))
            rows.append(dict(n=n, m=m, parity=parity, q=float(q),
                             k_sq=float(md.k_sq), zeros=zeros,
                             ok=zeros == m - 1))
        return rows

    # ── L² norms on the elliptic domain ─────────────────────────
    def _compute_norms(self):
        a = self.c_focal

        gl_x, gl_w = np.polynomial.legendre.leggauss(self.N_mu_quad)
        mu_nodes = 0.5 * self.mu_0 * (gl_x + 1)
        mu_weights = 0.5 * self.mu_0 * gl_w
        nu_nodes = np.linspace(0, 2 * np.pi, self.N_nu_quad, endpoint=False)
        dnu = 2 * np.pi / self.N_nu_quad

        # Jacobian h² = a²(sinh²μ + sin²ν) of the elliptic transform
        sinh2 = np.sinh(mu_nodes)**2
        sin2 = np.sin(nu_nodes)**2
        h2 = a**2 * (sinh2[:, None] + sin2[None, :])
        assert float(np.min(h2)) > 0.0, \
            "quadrature node with J <= 0 (focal degeneracy on the grid)"

        self._mu_quad = (mu_nodes, mu_weights)
        self._nu_quad = nu_nodes
        self._dnu = dnu

        for md in self.modes:
            n, q, parity = md.meta["n"], md.meta["q"], md.meta["parity"]
            R_vals, _, _, A_vals, _, _ = self._separated_value_deriv2(
                n, q, parity, mu_nodes, nu_nodes
            )
            psi2 = (R_vals[:, None] * A_vals[None, :]) ** 2
            norm_sq = np.sum(mu_weights[:, None] * self._dnu * psi2 * h2)
            md.norm = float(np.sqrt(max(norm_sq, 1e-30)))

    # ── Elliptic coord helpers ──────────────────────────────────
    def _spot_to_elliptic(self, S1, S2):
        x1, x2 = self.market.spot_to_x(S1, S2)
        z1, z2 = self.market.x_to_z(x1, x2)
        w = (z2 + 1j * z1) / self.c_focal
        ell = np.arccosh(w)
        return abs(np.real(ell)), np.imag(ell) % (2 * np.pi)

    def _elliptic_to_z(self, mu, nu):
        return (self.c_focal * np.sinh(mu) * np.sin(nu),
                self.c_focal * np.cosh(mu) * np.cos(nu))

    def _elliptic_to_spot(self, mu, nu):
        z1, z2 = self._elliptic_to_z(mu, nu)
        x1, x2 = self.market.z_to_x(z1, z2)
        return self.market.S1_0 * np.exp(x1), self.market.S2_0 * np.exp(x2)

    # ── ψ at an interior point ──────────────────────────────────
    def psi_at_spot(self, mode, S1, S2):
        mu, nu = self._spot_to_elliptic(S1, S2)
        if mu > self.mu_0:
            return 0.0
        return self._psi_mu_nu(mode, mu, nu)

    def _psi_mu_nu(self, mode, mu, nu):
        n, q, parity = mode.meta["n"], mode.meta["q"], mode.meta["parity"]
        Rv, _, _, Av, _, _ = self._separated_value_deriv2(
            n, q, parity, mu, nu
        )
        return Rv * Av

    def _dpsi_mu_nu(self, mode, mu, nu):
        n, q, parity = mode.meta["n"], mode.meta["q"], mode.meta["parity"]
        Rv, Rd, _, Av, Ad, _ = self._separated_value_deriv2(
            n, q, parity, mu, nu
        )
        return Rv, Rd, Av, Ad

    def mode_residual_certificates(self, mode, *, n_mu=16, n_nu=48):
        """Boundary and interior residuals of each mode."""
        n, q, parity = mode.meta["n"], mode.meta["q"], mode.meta["parity"]
        mu = np.linspace(0.02 * self.mu_0, 0.98 * self.mu_0, n_mu)
        nu = np.linspace(0, 2 * np.pi, n_nu, endpoint=False)
        Rv, _, Rdd, Av, _, Add = self._separated_value_deriv2(
            n, q, parity, mu, nu
        )
        Rb, _, _, _, _, _ = self._separated_value_deriv2(
            n, q, parity, self.mu_0, 0.0
        )

        psi = Rv[:, None] * Av[None, :]
        boundary_max = abs(float(Rb)) * float(np.max(np.abs(Av)))
        interior_max = float(np.max(np.abs(psi)))
        r_bc = boundary_max / max(interior_max, 1e-300)

        h2 = self.c_focal ** 2 * (
            np.sinh(mu)[:, None] ** 2 + np.sin(nu)[None, :] ** 2
        )
        lap_num = Rdd[:, None] * Av[None, :] + Rv[:, None] * Add[None, :]
        residual = lap_num / h2 + mode.k_sq * psi
        num = np.sqrt(np.sum(h2 * residual ** 2))
        den = np.sqrt(np.sum(h2 * psi ** 2))
        eps = float(num / max(den, 1e-300))
        return dict(r_bc=float(r_bc), eps=eps, char_value=float(Rb))

    def _boundary_residual_certificate(self, mode):
        """Boundary residual of each mode."""
        cert = self.mode_residual_certificates(mode)
        return cert["r_bc"], cert["char_value"]

    # ── Collocation grid (for SVD projection) ───────────────────
    def collocation_grid_spot(self):
        """Flat arrays of spot prices at the (μ, ν) quadrature nodes."""
        mu_nodes, mu_weights = self._mu_quad
        nu_nodes = self._nu_quad
        mu_flat = np.repeat(mu_nodes, len(nu_nodes))
        nu_flat = np.tile(nu_nodes, len(mu_nodes))
        S1_flat = np.empty_like(mu_flat)
        S2_flat = np.empty_like(mu_flat)
        for i in range(len(mu_flat)):
            s1, s2 = self._elliptic_to_spot(mu_flat[i], nu_flat[i])
            S1_flat[i], S2_flat[i] = s1, s2
        # Cache natural-coord versions so design_matrix_at_spot can skip the
        # round-trip spot→μ,ν for the common path.
        self._colloc_mu = mu_flat
        self._colloc_nu = nu_flat
        self._colloc_mu_weights = np.repeat(mu_weights, len(nu_nodes))
        return S1_flat, S2_flat

    def design_matrix_at_spot(self, S1_arr, S2_arr, *, tensorised=None):
        """A[i,k] = ψ_k(S1_i,S2_i) / ‖ψ_k‖, using cached μ,ν when available."""
        n_pts = len(S1_arr)
        n_modes = len(self.modes)
        A = np.zeros((n_pts, n_modes))
        on_colloc_grid = (hasattr(self, "_colloc_mu")
                          and len(self._colloc_mu) == n_pts)
        if tensorised is None:
            tensorised = self.tensor_design
        if tensorised and on_colloc_grid and self.backend == "tridiagonal":
            mu_nodes, _ = self._mu_quad
            nu_nodes = self._nu_quad
            for k, md in enumerate(self.modes):
                norm = md.norm if md.norm > 1e-15 else 1.0
                n, q, parity = md.meta["n"], md.meta["q"], md.meta["parity"]
                Rv, _, _, Av, _, _ = self._separated_value_deriv2(
                    n, q, parity, mu_nodes, nu_nodes
                )
                # Same op order as the flattened path: (R·A) then /norm.
                A[:, k] = np.multiply.outer(Rv, Av).ravel() / norm
            return A
        if on_colloc_grid:
            mu_flat = self._colloc_mu
            nu_flat = self._colloc_nu
        else:
            mu_flat = np.empty(n_pts)
            nu_flat = np.empty(n_pts)
            for i in range(n_pts):
                mu_flat[i], nu_flat[i] = self._spot_to_elliptic(
                    S1_arr[i], S2_arr[i])
        for k, md in enumerate(self.modes):
            norm = md.norm if md.norm > 1e-15 else 1.0
            n, q, parity = md.meta["n"], md.meta["q"], md.meta["parity"]
            Rv, _, _, Av, _, _ = self._separated_value_deriv2(
                n, q, parity, mu_flat, nu_flat
            )
            A[:, k] = (Rv * Av) / norm
        return A

    # ── Convenience for the spectral solver's analytic-delta path ──
    def eval_with_derivatives(self, S1, S2, tau_eff):
        """Return u, ∂u/∂μ, ∂u/∂ν, ∂u/∂τ (sums over modes, normalised)."""
        mu, nu = self._spot_to_elliptic(S1, S2)
        if mu > self.mu_0:
            return 0.0, 0.0, 0.0, 0.0
        u = du_dmu = du_dnu = theta_sum = 0.0
        for md in self.modes:
            c = md.coeff
            if abs(c) < 1e-15:
                continue
            Rv, Rd, Av, Ad = self._dpsi_mu_nu(md, mu, nu)
            norm = md.norm if md.norm > 1e-15 else 1.0
            tf = md.decay(tau_eff)
            psi_n = (Rv * Av) / norm
            u         += c * psi_n * tf
            du_dmu    += c * (Rd * Av) / norm * tf
            du_dnu    += c * (Rv * Ad) / norm * tf
            theta_sum += c * psi_n * (-md.k_sq / 2) * tf
        return u, du_dmu, du_dnu, theta_sum

    def mu_nu_at_spot(self, S1, S2):
        return self._spot_to_elliptic(S1, S2)
