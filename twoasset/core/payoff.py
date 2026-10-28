"""Payoff functions."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Payoff(ABC):
    """Abstract payoff: takes spot prices, returns the final cashflow."""

    boundary_compatible: bool = True

    @abstractmethod
    def __call__(self, S1, S2):
        ...

    # Subclasses can override with a tighter human-readable name
    def __repr__(self):
        return f"{type(self).__name__}()"


# ── Vanilla / contractual payoffs ──────────────────────────────────

class BasketCall(Payoff):
    """max(w₁·S₁ + w₂·S₂ − K, 0)."""
    def __init__(self, K: float, w1: float = 1.0, w2: float = 1.0):
        self.K = K
        self.w1 = w1
        self.w2 = w2

    def __call__(self, S1, S2):
        return np.maximum(self.w1 * S1 + self.w2 * S2 - self.K, 0.0)

    def __repr__(self):
        return f"BasketCall(K={self.K}, w={self.w1},{self.w2})"


class Spread(Payoff):
    """max(w₁·S₁ − w₂·S₂ − K, 0)."""
    def __init__(self, K: float, w1: float = 1.0, w2: float = 1.0):
        self.K = K
        self.w1 = w1
        self.w2 = w2

    def __call__(self, S1, S2):
        return np.maximum(self.w1 * S1 - self.w2 * S2 - self.K, 0.0)


class WorstOf(Payoff):
    """max(min(S₁/S₁⁰, S₂/S₂⁰)·S₁⁰ − K, 0) — return-normalised worst-of call."""
    def __init__(self, K: float, S1_0: float, S2_0: float):
        self.K = K
        self.S1_0 = S1_0
        self.S2_0 = S2_0

    def __call__(self, S1, S2):
        worst = np.minimum(S1 / self.S1_0, S2 / self.S2_0)
        return np.maximum(worst * self.S1_0 - self.K, 0.0)


class DoubleNoTouch(Payoff):
    """Fixed notional paid at expiry if the path survives inside the domain."""

    # Constant on the domain: the whole payoff is the jump at the barrier.
    boundary_compatible = False

    def __init__(self, notional: float = 1.0):
        self.notional = notional

    def __call__(self, S1, S2):
        return self.notional * np.ones_like(np.asarray(S1, dtype=float))

    def __repr__(self):
        return f"DoubleNoTouch(N={self.notional})"


class UnitPayoff(Payoff):
    """Terminal value ≡ 1 everywhere — the survival / no-touch bond payoff."""

    boundary_compatible = False

    def __init__(self, notional: float = 1.0):
        self.notional = notional

    def __call__(self, S1, S2):
        return self.notional * np.ones_like(np.asarray(S1, dtype=float))

    def __repr__(self):
        return f"UnitPayoff(N={self.notional})"


class Digital(Payoff):
    """Notional · 𝟙[w₁S₁ + w₂S₂ ≥ K]."""
    def __init__(self, K: float, notional: float = 1.0,
                 w1: float = 1.0, w2: float = 1.0):
        self.K = K
        self.notional = notional
        self.w1 = w1
        self.w2 = w2

    def __call__(self, S1, S2):
        return self.notional * np.where(
            self.w1 * S1 + self.w2 * S2 >= self.K, 1.0, 0.0
        )


class SmoothBump(Payoff):
    """Smooth polynomial bump in log-return space:"""
    def __init__(self, S1_0: float, S2_0: float,
                 r2_max: float = 0.25, amp: float = 0.5):
        self.S1_0 = S1_0
        self.S2_0 = S2_0
        self.r2_max = r2_max
        self.amp = amp

    def __call__(self, S1, S2):
        x1 = np.log(S1 / self.S1_0)
        x2 = np.log(S2 / self.S2_0)
        r2 = x1**2 + x2**2
        return self.amp * np.maximum(self.r2_max - 0.3 * r2, 0.0)


class TriangularTent(Payoff):
    """Radial tent in log-return space: amp · max(R − √(x₁² + x₂²), 0)."""
    def __init__(self, S1_0: float, S2_0: float, R: float, amp: float = 1.0):
        self.S1_0 = S1_0
        self.S2_0 = S2_0
        self.R = R
        self.amp = amp

    def __call__(self, S1, S2):
        x1 = np.log(S1 / self.S1_0)
        x2 = np.log(S2 / self.S2_0)
        r = np.sqrt(x1**2 + x2**2)
        return self.amp * np.maximum(self.R - r, 0.0)


class PureEigenfunction(Payoff):
    """Wrap a single basis eigenfunction as a payoff."""
    def __init__(self, basis, mode, market):
        self.basis = basis
        self.mode = mode
        self.market = market

    def __call__(self, S1, S2):
        S1_arr = np.atleast_1d(np.asarray(S1, dtype=float))
        S2_arr = np.atleast_1d(np.asarray(S2, dtype=float))
        # Broadcast S1, S2 to a common shape.
        S1_b, S2_b = np.broadcast_arrays(S1_arr, S2_arr)
        out = np.empty_like(S1_b, dtype=float)
        for i, (a, b) in enumerate(zip(S1_b.ravel(), S2_b.ravel())):
            out.ravel()[i] = self.basis.psi_at_spot(self.mode, float(a), float(b))
        if np.isscalar(S1) and np.isscalar(S2):
            return float(out.ravel()[0])
        return out
