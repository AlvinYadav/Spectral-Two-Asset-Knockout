"""Production numerical settings (mode counts, smoothing) of the six contracts, and
shared paths."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = str(_REPO_ROOT / "validation" / "artifacts")


def _resolve_figures_dir() -> str:
    """Where rendered figures (PDF/PNG) are written."""
    env = os.environ.get("SPECTRAL_FIGURES_DIR")
    if env:
        return env
    sibling = (_REPO_ROOT.parent
               / "Paper-Pricing_Two_Asset_Options_with_Spectral_Methods"
               / "_shared" / "Figures-Tables")
    if sibling.is_dir():
        return str(sibling)
    return str(_REPO_ROOT / "validation" / "figures")


PAPER_FIGURES_DIR = _resolve_figures_dir()

ALL_CONFIGS = (
    "default", "tight_barrier", "mahalanobis", "dnt", "spread", "worst_of",
)

MATHIEU_BESSEL_CONFIGS = ("default", "tight_barrier", "mahalanobis")
FOURIER_CONFIGS = ("dnt", "spread", "worst_of")

OPTIMAL = {
    "default":       {"smooth_tau": 0.000, "N_modes": 25},
    "tight_barrier": {"smooth_tau": 0.000, "N_modes": 25},
    "mahalanobis":   {"smooth_tau": 0.000, "N_modes": 25},
    "dnt":           {"smooth_tau": 0.000, "N_modes": 20},
    "spread":        {"smooth_tau": 0.000, "N_modes": 50},
    "worst_of":      {"smooth_tau": 0.000, "N_modes": 32},
}


def build_solver(cfg_name: str, *, config_dir: str = "configs"):
    """Assemble + fit + project a SpectralSolver for `cfg_name` at its optimal
    settings."""
    import copy
    import tomllib
    from pathlib import Path

    from ..cli import assemble
    from .spectral import SpectralSolver

    opt = OPTIMAL[cfg_name]
    cfg_path = Path(config_dir) / f"{cfg_name}.toml"
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    N_modes = opt["N_modes"]
    N_quad = max(120, N_modes * 4)
    cfg2 = copy.deepcopy(cfg)
    cfg2.setdefault("spectral", {})["N_modes"] = N_modes
    cfg2["spectral"]["N_quad"] = N_quad

    market, payoff, domain, basis, _ = assemble(cfg2)
    smooth_tau = opt["smooth_tau"]
    solver = SpectralSolver(
        market=market, payoff=payoff, domain=domain, basis=basis,
        smooth=(smooth_tau > 0),
        smooth_tau_frac=max(smooth_tau, 1e-12),
    )
    solver.fit()
    solver.project()
    return solver, market, payoff, domain, basis


def load_mc_reference(cfg_name: str, validation_dir: str = "validation"):
    """Return (mc_mean, mc_sem) from validation/{cfg_name}.toml, or (None, None) if
    that file doesn't exist yet."""
    import tomllib
    from pathlib import Path

    vpath = Path(validation_dir) / f"{cfg_name}.toml"
    if not vpath.exists():
        return None, None
    with open(vpath, "rb") as f:
        v = tomllib.load(f)
    mc = v.get("mc", {})
    return mc.get("mean"), mc.get("sem")
