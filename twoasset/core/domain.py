"""Survival domains in log-return space."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


def _bgk_delta_z(monitoring_dt: float | None) -> float:
    """δ_z = 0.5826 · √Δt, or 0.0 if continuous monitoring (no correction)."""
    if monitoring_dt is None:
        return 0.0
    return 0.5826 * np.sqrt(monitoring_dt)


class SurvivalDomain(ABC):
    """Knockout survival region in x-space."""

    @abstractmethod
    def contains(self, x1, x2) -> np.ndarray:
        ...

    def contains_spot(self, S1, S2, market) -> np.ndarray:
        x1, x2 = market.spot_to_x(S1, S2)
        return self.contains(x1, x2)

    @abstractmethod
    def default_basis_name(self) -> str:
        ...


# ── Euclidean circle in x-space → ellipse in z-space ─────────────

@dataclass
class EuclideanCircle(SurvivalDomain):
    """Circular barrier in log-return space: x₁² + x₂² ≤ R²."""
    R: float
    monitoring_dt: float | None = None

    def contains(self, x1, x2):
        x1 = np.asarray(x1); x2 = np.asarray(x2)
        return x1**2 + x2**2 <= self.R**2

    def default_basis_name(self):
        return "mathieu"

    def geometry(self, market):
        """Return (A_semi, B_semi, c_focal, mu_0, eccentricity)."""
        A = self.R / np.sqrt(market.lam1)   # minor semi-axis (larger λ → shorter)
        B = self.R / np.sqrt(market.lam2)   # major semi-axis
        delta_z = _bgk_delta_z(self.monitoring_dt)
        A = A + delta_z
        B = B + delta_z
        c = np.sqrt(B**2 - A**2) if B > A else 1e-12
        mu_0 = np.arctanh(A / B)
        ecc = np.sqrt(1 - (A / B)**2)
        return dict(A_semi=A, B_semi=B, c_focal=c, mu_0=mu_0,
                    eccentricity=ecc)


# ── Mahalanobis ellipse in x-space → circle in z-space ───────────

@dataclass
class MahalanobisEllipse(SurvivalDomain):
    """Mahalanobis-distance barrier: xᵀ Σ⁻¹ x ≤ R²."""
    R: float
    monitoring_dt: float | None = None

    def contains(self, x1, x2):
        # Work in x-space via the inverse covariance.  Market not stored
        # so callers should prefer contains_spot/contains_z via a market.
        raise NotImplementedError(
            "MahalanobisEllipse.contains requires a market; use "
            "contains_spot(S1, S2, market) or contains_z."
        )

    def contains_spot(self, S1, S2, market):
        x1, x2 = market.spot_to_x(np.asarray(S1), np.asarray(S2))
        z1, z2 = market.x_to_z(x1, x2)
        return z1**2 + z2**2 <= self.R**2

    def contains_z(self, z1, z2):
        z1 = np.asarray(z1); z2 = np.asarray(z2)
        return z1**2 + z2**2 <= self.R**2

    def default_basis_name(self):
        return "bessel"

    def geometry(self, market):
        """Circle of radius R in z-space — nothing market-dependent."""
        delta_z = _bgk_delta_z(self.monitoring_dt)
        return dict(radius=self.R + delta_z)


# ── Rectangle in x-space ─────────────────────────────────────────

@dataclass
class Rectangle(SurvivalDomain):
    """Axis-aligned rectangle in x-space: [x1_lo, x1_hi] × [x2_lo, x2_hi]."""
    x1_lo: float
    x1_hi: float
    x2_lo: float
    x2_hi: float
    monitoring_dt: float | None = None

    @property
    def W1(self) -> float:
        return self.x1_hi - self.x1_lo

    @property
    def W2(self) -> float:
        return self.x2_hi - self.x2_lo

    def contains(self, x1, x2):
        x1 = np.asarray(x1); x2 = np.asarray(x2)
        return ((x1 >= self.x1_lo) & (x1 <= self.x1_hi)
                & (x2 >= self.x2_lo) & (x2 <= self.x2_hi))

    def default_basis_name(self):
        return "fourier"

    def geometry(self, market):
        delta_z = _bgk_delta_z(self.monitoring_dt)
        return dict(W1=self.W1 + 2 * delta_z, W2=self.W2 + 2 * delta_z,
                    x1_lo=self.x1_lo - delta_z, x2_lo=self.x2_lo - delta_z)
