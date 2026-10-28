"""Abstract eigenbasis interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class EigenMode:
    """A single Dirichlet eigenmode of the Laplacian on the survival domain."""
    idx: tuple          # implementation-defined index tuple, e.g. (n, m, parity)
    k_sq: float
    norm: float = 1.0
    coeff: float = 0.0
    meta: dict = None   # basis-specific extras (q, parity, ...) stored here

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}

    def decay(self, tau: float) -> float:
        return np.exp(-0.5 * self.k_sq * tau)


class EigenBasis(ABC):
    """Abstract Dirichlet eigenbasis on a survival domain."""

    modes: list          # set by .fit()
    market = None
    domain = None

    # ── Setup ────────────────────────────────────────────────────
    @abstractmethod
    def fit(self, market, domain) -> "EigenBasis":
        """Find eigenvalues, compute norms."""
        ...

    # ── Evaluation ───────────────────────────────────────────────
    @abstractmethod
    def psi_at_spot(self, mode: EigenMode, S1: float, S2: float) -> float:
        """ψ(S1, S2) — eigenfunction evaluated at a spot price."""
        ...

    # ── Projection via SVD collocation ───────────────────────────
    @abstractmethod
    def collocation_grid_spot(self):
        """Return arrays (S1_colloc, S2_colloc) of collocation spots."""
        ...

    @abstractmethod
    def design_matrix_at_spot(self, S1_arr, S2_arr):
        """(N_pts × N_modes) matrix A[i,k] = ψ_k(S1_i, S2_i) / ‖ψ_k‖."""
        ...

    # ── Price-time evaluation ───────────────────────────────────
    def eval_expansion(self, S1, S2, tau_eff: float) -> float:
        """Σ_k c_k · (ψ_k(S)/‖ψ_k‖) · exp(−½ k² τ_eff)."""
        total = 0.0
        for md in self.modes:
            if abs(md.coeff) < 1e-15:
                continue
            psi = self.psi_at_spot(md, S1, S2)
            total += md.coeff * (psi / md.norm) * md.decay(tau_eff)
        return total

    # ── Diagnostics ──────────────────────────────────────────────
    def n_modes(self) -> int:
        return len(self.modes)

    def active_modes(self, tol: float = 1e-6):
        """Modes with non-negligible norm — used to prune degenerate ones."""
        return [m for m in self.modes if m.norm > tol]


def gram_dedupe(modes: list, psi_flat: np.ndarray, quad_weight_flat: np.ndarray,
                *, dup_threshold: float = 0.999, cond_warn: float = 1e4,
                label: str = "") -> tuple:
    """Post-scan Gram-matrix validation, shared across bases."""
    n = len(modes)
    weighted = psi_flat * quad_weight_flat[None, :]
    G = weighted @ psi_flat.T

    keep = [True] * n
    dropped = []
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            val = abs(G[i, j])
            if val > dup_threshold:
                keep[j] = False
                dropped.append((i, j, val))

    kept_idx = [k for k in range(n) if keep[k]]
    kept_modes = [modes[k] for k in kept_idx]
    G_kept = G[np.ix_(kept_idx, kept_idx)]
    max_offdiag = float(np.max(np.abs(G_kept - np.diag(np.diag(G_kept))))) if len(kept_idx) > 1 else 0.0
    cond = float(np.linalg.cond(G_kept)) if len(kept_idx) > 1 else 1.0

    import os
    sig = (label, n, len(dropped))
    verbose = os.environ.get("TWOASSET_VERBOSE_GRAM") == "1"
    if verbose or sig not in _GRAM_LOGGED:
        _GRAM_LOGGED.add(sig)
        tag = f"[{label}] " if label else ""
        if dropped:
            print(f"{tag}WARNING: dropped {len(dropped)} near-duplicate "
                  f"mode(s) (|<Psi_i,Psi_j>| > {dup_threshold}):")
            for i, j, val in dropped:
                print(f"  mode {j} {modes[j].meta} duplicates mode {i} "
                      f"{modes[i].meta}: |<Psi_i,Psi_j>|={val:.5f}")
        print(f"{tag}Gram matrix health: {len(kept_idx)}/{n} modes retained, "
              f"max off-diagonal {max_offdiag:.4f}, cond(G)={cond:.2e}"
              + (f" -- exceeds cond_warn={cond_warn:.0e}" if cond > cond_warn else ""))
    dropped_meta = [(modes[i].meta, modes[j].meta, val) for i, j, val in dropped]
    return kept_modes, G_kept, dropped_meta


_GRAM_LOGGED: set = set()
