"""Spectral pricer for two-asset European knockout options."""

from .core.market import GBMMarket
from .core.payoff import (
    Payoff, BasketCall, Spread, WorstOf, Digital,
    SmoothBump, TriangularTent, PureEigenfunction,
)
from .core.domain import (
    SurvivalDomain, EuclideanCircle, MahalanobisEllipse, Rectangle,
)
from .bases.base import EigenBasis, EigenMode
from .bases.mathieu import MathieuBasis
from .bases.bessel import BesselBasis
from .bases.fourier import FourierBasis
from .solvers.spectral import SpectralSolver
from .solvers.fdm import fdm_price, fdm_smooth
from .solvers.mc import mc_price

__all__ = [
    "GBMMarket",
    "Payoff", "BasketCall", "Spread", "WorstOf", "Digital",
    "SmoothBump", "TriangularTent", "PureEigenfunction",
    "SurvivalDomain", "EuclideanCircle", "MahalanobisEllipse", "Rectangle",
    "EigenBasis", "EigenMode",
    "MathieuBasis", "BesselBasis", "FourierBasis",
    "SpectralSolver",
    "fdm_price", "fdm_smooth", "mc_price",
]
