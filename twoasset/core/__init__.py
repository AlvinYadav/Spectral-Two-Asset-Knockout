"""Problem-specification layer: market, payoff, domain."""

from .market import GBMMarket
from .payoff import (
    Payoff, BasketCall, Spread, WorstOf, Digital,
    SmoothBump, TriangularTent, PureEigenfunction,
)
from .domain import SurvivalDomain, EuclideanCircle, MahalanobisEllipse, Rectangle

__all__ = [
    "GBMMarket",
    "Payoff", "BasketCall", "Spread", "WorstOf", "Digital",
    "SmoothBump", "TriangularTent", "PureEigenfunction",
    "SurvivalDomain", "EuclideanCircle", "MahalanobisEllipse", "Rectangle",
]
