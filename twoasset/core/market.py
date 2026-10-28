"""Two-asset geometric Brownian motion market."""

from dataclasses import dataclass, field
import numpy as np

from .market_base import Market


@dataclass
class GBMMarket(Market):
    """Two-asset GBM under the risk-neutral measure."""

    S1_0: float
    S2_0: float
    sigma1: float
    sigma2: float
    rho: float
    r: float
    T: float

    # ── Derived (filled in __post_init__) ───────────────────────────
    Sigma: np.ndarray = field(init=False, repr=False)
    lam1: float = field(init=False)             # larger eigenvalue of Σ
    lam2: float = field(init=False)             # smaller eigenvalue of Σ
    P: np.ndarray = field(init=False, repr=False)  # eigenvector matrix
    theta: float = field(init=False)            # principal-axis angle

    beta1: float = field(init=False)
    beta2: float = field(init=False)
    gamma: float = field(init=False)

    M_xz: np.ndarray = field(init=False, repr=False)  # x → z
    M_zx: np.ndarray = field(init=False, repr=False)  # z → x

    # PDE coefficients (useful for FDM/Galerkin in x-space)
    a11: float = field(init=False)
    a12: float = field(init=False)
    a22: float = field(init=False)

    def __post_init__(self):
        s1, s2, rho = self.sigma1, self.sigma2, self.rho
        self.Sigma = np.array([[s1**2, rho*s1*s2],
                               [rho*s1*s2, s2**2]])

        evals, evecs = np.linalg.eigh(self.Sigma)
        idx = evals.argsort()[::-1]
        self.lam1, self.lam2 = evals[idx]
        self.P = evecs[:, idx]
        self.theta = 0.5 * np.arctan2(2*rho*s1*s2, s1**2 - s2**2)

        alpha = np.array([self.r - 0.5*s1**2, self.r - 0.5*s2**2])
        Sigma_inv = np.linalg.inv(self.Sigma)
        beta = -Sigma_inv @ alpha
        self.beta1, self.beta2 = beta
        self.gamma = -0.5 * alpha @ Sigma_inv @ alpha - self.r

        D_inv_sqrt = np.diag([1.0/np.sqrt(self.lam1), 1.0/np.sqrt(self.lam2)])
        self.M_xz = D_inv_sqrt @ self.P.T
        self.M_zx = np.linalg.inv(self.M_xz)

        self.a11 = 0.5 * s1**2
        self.a12 = rho * s1 * s2
        self.a22 = 0.5 * s2**2

    # ── Build from a parsed TOML config ───────────────────────────
    @classmethod
    def from_config(cls, cfg):
        fin = cfg["financial"]
        return cls(
            S1_0=fin["S1_0"], S2_0=fin["S2_0"],
            sigma1=fin["sigma1"], sigma2=fin["sigma2"], rho=fin["rho"],
            r=fin["r"], T=fin["T"],
        )

    def summary(self) -> str:
        return (
            f"λ₁={self.lam1:.6f}  λ₂={self.lam2:.6f}  θ={np.rad2deg(self.theta):+.1f}°\n"
            f"β=({self.beta1:+.4f},{self.beta2:+.4f})  γ={self.gamma:+.6f}"
        )
