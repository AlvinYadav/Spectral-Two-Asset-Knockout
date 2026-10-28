"""Earlier Mathieu functions based on SciPy, kept for comparison (Remark 3.1, Table
6.6)."""

import numpy as np
from scipy.special import mathieu_cem, mathieu_sem
from functools import lru_cache

# ── Internal: Fourier coefficient extraction ──────────────────────────

_cache = {}

def _fourier_key(n, q, parity, N):
    return (n, round(q, 12), parity, N)


def _get_fourier_coeffs(n, q, parity, N):
    """Extract Fourier coefficients of ce_n or se_n via DFT."""
    key = _fourier_key(n, q, parity, N)
    if key in _cache:
        return _cache[key]

    n_pts = 2 * N + 1
    nu = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    nu_deg = np.rad2deg(nu)

    if parity == 'even':
        vals = np.array([mathieu_cem(n, q, d)[0] for d in nu_deg])
        coeffs = np.zeros(N + 1)
        for r in range(N + 1):
            coeffs[r] = (2.0 / n_pts) * np.sum(vals * np.cos(r * nu))
        coeffs[0] /= 2
    else:
        vals = np.array([mathieu_sem(n, q, d)[0] for d in nu_deg])
        coeffs = np.zeros(N + 1)
        for r in range(1, N + 1):
            coeffs[r] = (2.0 / n_pts) * np.sum(vals * np.sin(r * nu))

    # Truncate negligible tails (prevents cosh overflow for large r·μ)
    mx = np.max(np.abs(coeffs))
    if mx > 0:
        coeffs[np.abs(coeffs) < 1e-14 * mx] = 0.0
    nz = np.nonzero(coeffs)[0]
    coeffs = coeffs[:nz[-1] + 1] if len(nz) > 0 else coeffs[:1]

    _cache[key] = coeffs
    return coeffs


# ── Angular Mathieu functions (SciPy wrappers) ───────────────────────

def ce(n, q, nu, N=50):
    """Even angular Mathieu function ce_n(ν, q)."""
    return mathieu_cem(n, q, np.rad2deg(nu))[0]

def se(n, q, nu, N=50):
    """Odd angular Mathieu function se_n(ν, q)."""
    return mathieu_sem(n, q, np.rad2deg(nu))[0]

def ce_deriv(n, q, nu, N=50):
    """d/dν ce_n(ν, q)."""
    return mathieu_cem(n, q, np.rad2deg(nu))[1]

def se_deriv(n, q, nu, N=50):
    """d/dν se_n(ν, q)."""
    return mathieu_sem(n, q, np.rad2deg(nu))[1]


# ── Radial (modified) Mathieu functions via Fourier series ───────────

def Ce(n, q, mu, N=50):
    """Even radial Mathieu function: Ce_n(μ, q) = Σ A_r cosh(rμ)."""
    c = _get_fourier_coeffs(n, q, 'even', N)
    result = c[0]
    for r in range(1, len(c)):
        if c[r] != 0:
            result += c[r] * np.cosh(r * mu)
    return result

def Se(n, q, mu, N=50):
    """Odd radial Mathieu function: Se_n(μ, q) = Σ B_r sinh(rμ)."""
    c = _get_fourier_coeffs(n, q, 'odd', N)
    result = 0.0
    for r in range(1, len(c)):
        if c[r] != 0:
            result += c[r] * np.sinh(r * mu)
    return result

def Ce_deriv(n, q, mu, N=50):
    """d/dμ Ce_n(μ, q) = Σ r·A_r sinh(rμ)."""
    c = _get_fourier_coeffs(n, q, 'even', N)
    result = 0.0
    for r in range(1, len(c)):
        if c[r] != 0:
            result += r * c[r] * np.sinh(r * mu)
    return result

def Se_deriv(n, q, mu, N=50):
    """d/dμ Se_n(μ, q) = Σ r·B_r cosh(rμ)."""
    c = _get_fourier_coeffs(n, q, 'odd', N)
    result = 0.0
    for r in range(1, len(c)):
        if c[r] != 0:
            result += r * c[r] * np.cosh(r * mu)
    return result


def clear_cache():
    """Clear the Fourier coefficient cache."""
    _cache.clear()
