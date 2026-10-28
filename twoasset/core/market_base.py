"""Abstract market interface."""

from __future__ import annotations
from abc import ABC
import numpy as np


class Market(ABC):
    """Abstract two-asset market."""

    # ── Coordinate helpers ────────────────────────────────────────────
    def spot_to_x(self, S1, S2):
        """Spot → log-return (x₁, x₂) = (log S₁/S₁⁰, log S₂/S₂⁰)."""
        return np.log(S1 / self.S1_0), np.log(S2 / self.S2_0)

    def x_to_spot(self, x1, x2):
        """Log-return → spot."""
        return self.S1_0 * np.exp(x1), self.S2_0 * np.exp(x2)

    def x_to_z(self, x1, x2):
        """Log-return → whitened coordinates z = M_xz x."""
        z = self.M_xz @ np.array([x1, x2])
        return z[0], z[1]

    def z_to_x(self, z1, z2):
        """Whitened → log-return x = M_zx z."""
        x = self.M_zx @ np.array([z1, z2])
        return x[0], x[1]

    def gauge_factor(self, S1, S2, tau):
        """exp(β·log S + γτ) = S₁^β₁ S₂^β₂ e^(γτ)."""
        return np.exp(
            self.beta1 * np.log(S1) + self.beta2 * np.log(S2) + self.gamma * tau
        )
