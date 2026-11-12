# Spectral pricing of two-asset knockout options

Code and evidence for

> A. Yadav and S. Petrat, *Geometry-Adapted Spectral Pricing of Two-Asset Knockout
> Options*, preprint, 2026.

Two-asset European knockout options under constant-coefficient Black–Scholes dynamics
are priced by eigenfunction expansion. After a change of variables the pricing
equation becomes the heat equation on a bounded domain, and the shape of the
transformed barrier selects the basis: Fourier–Galerkin modes on rectangles, Bessel
modes on discs and Mathieu modes on ellipses. Each mode evolves by an exact
exponential factor, so the price and its spot Greeks at any spot are a finite sum.
Finite differences in whitened coordinates (Crank–Nicolson with Shortley–Weller
boundaries) and bridge-corrected Monte Carlo are included as independent benchmarks.

## Installation

Python 3.11 or later.

```bash
pip install -r requirements.txt
```

`requirements-lock.txt` pins the exact versions used for the paper (Python 3.11.9).

## Quick start

```python
from twoasset.solvers._optimal import build_solver

solver, *_ = build_solver("default")      # one of the six contracts below
print(solver.price(), solver.greeks())
```

`build_solver` applies the production settings of the paper (mode counts, no payoff
smoothing), which are kept in `twoasset/solvers/_optimal.py`. The command-line
interface `python -m twoasset.cli configs/default.toml --no-mc --no-fdm --no-smooth`
reads a configuration file as written.

## Layout

```
twoasset/
  core/                      Market, payoffs, survival domains
  bases/                     Mathieu, Bessel and Fourier-Galerkin eigenbases
  solvers/                   Spectral solver, finite differences, Monte Carlo
  mathieu_backend.py         Mathieu functions from the three-term recurrence
  projection_comparison.py   Comparison of projection measures (Table 6.5)
  holdout_eval.py            Evaluation of a sealed holdout
legacy/                      SciPy-based Mathieu functions, kept for comparison
configs/                     The six contracts and the sealed holdouts
validation/                  Scripts behind the paper's tables, and their outputs
figures/                     Scripts for Figures 2.1 and 3.1, and an earlier cost figure
tests/                       Tests
```

The six contracts are `dnt`, `worst_of`, `spread` (rectangles, Fourier–Galerkin),
`mahalanobis` (disc, Bessel), and `default` and `tight_barrier` (ellipses, Mathieu).

## Reproducing the paper

Run the scripts from the repository root with `PYTHONPATH=.`. Data are written to
`validation/artifacts/`, figures as listed below.

| Paper | Script | Output |
|---|---|---|
| Figure 2.1 | `figures/geometry_x_to_z.py` | `p1_geometry_x_to_z.pdf` in the working directory |
| Figure 3.1 | `figures/elliptic_grid.py` | `p1_elliptic_grid.pdf` in the working directory |
| Remark 3.1 | `validation/paper1_scipy_mathieu.py` | `paper1_scipy_mathieu.json` |
| Quadrature refinement (Sections 3.4 and 6.5) | `validation/paper1_accuracy_checks.py quad` | `paper1_accuracy_quad.json` |
| Finite-difference benchmark (Section 5.1): disc, M-matrix, space and time refined separately, Cholesky against eigen whitening | `validation/paper1_zfdm.py z1 z2 sep levels` | `paper1_zfdm_<section>.json` |
| Remark 5.1 (log-return finite differences at ρ = 0) | `validation/paper1_zfdm.py rho0` | `paper1_zfdm_rho0.json` |
| Monte Carlo references | `validation/paper1_mc.py` | `paper1_mc.json`, `paper1_mc_parts/` |
| Monitoring bias of the Monte Carlo benchmark (Section 5.2) | `validation/paper1_mc_bias.py run <contract>`, then `merge` | `paper1_mc_bias.json`, `paper1_mc_bias_parts/` |
| Table 6.1 (a) | `validation/paper1_price_table.py` | `paper1_price_table.json` |
| Table 6.1 (b), and the off-centre spots (Section 6.1) | `validation/paper1_zfdm.py levels xgreeks`; the spots are those of `validation/paper1_accuracy_checks.py offcentre <contract>` | `paper1_zfdm_levels.json`, `paper1_zfdm_xgreeks.json`, `paper1_accuracy_offcentre_<contract>.json` |
| Mode counts (Section 6.1) | `validation/paper1_modes.py` | `paper1_modes.json` |
| Quadrature doubled on the curved barriers (Section 6.1) | `validation/paper1_cost2.py run`, then `report` (`spectral_refined_quadrature`) | `paper1_cost2_report.json` |
| Remark 6.1 | `validation/paper1_accuracy_checks.py fquad rho0` | `paper1_accuracy_fquad.json`, `paper1_accuracy_rho0.json` |
| Spread call with `K = 0` against Margrabe (Remark 6.1) | `validation/paper1_accuracy_checks.py margrabe` | `paper1_accuracy_margrabe.json` |
| Truncated domain of the worst-of call (Remark 6.1) | `validation/paper1_onefactor.py truncation` | `paper1_onefactor_truncation.json` |
| Table 6.2 (spectral rows), Figure 6.1 | `validation/paper1_greeks.py` | `paper1_greeks.json` |
| Table 6.2 (finite-difference rows), fixed-barrier vegas (Section 6.2) | `validation/paper1_zfdm.py levels xgreeks market xmarket` | `paper1_zfdm_<section>.json` |
| Table 6.3 | `validation/paper1_cost2.py run`, then `matched`, then `report` | `paper1_cost2.json`, `paper1_cost2_matched.json`, `paper1_cost2_report.json` |
| Table 6.4 | `validation/paper1_evidence.py`; finite-difference column from `paper1_zfdm.py levels xgreeks` | `paper1_evidence.json` |
| Table 6.5 | `python -m twoasset.projection_comparison` | `validation/projection_comparison.json` |
| Table 6.6 | `python -m twoasset.holdout_eval` | `validation/holdout_rerun.json` |
| Third grid for holdouts B and C (Section 6.5) | `validation/paper1_accuracy_checks.py holdouts` | `paper1_accuracy_holdouts.json` |
| Table 6.7 (largest change over the contracts of each group), Section 6.6 | `validation/paper1_accuracy_checks.py conv <contract>` | `paper1_accuracy_conv_<contract>.json` |
| The maturity T/4 against finite differences (Section 6.6) | `validation/paper1_zfdm.py short` | `paper1_zfdm_short.json` |
| Rectangle gammas against finite differences (Section 6.6) | `validation/paper1_accuracy_checks.py conv_fd <contract>` | `paper1_accuracy_conv_fd_<contract>.json`, and `..._T.json` on the rectangles |
| Modes per side on the rectangles (Section 6.6) | `validation/paper1_accuracy_checks.py conv_modes` | `paper1_accuracy_conv_modes.json` |
| Table A.2 | `tests/test_mathieu_backend.py` | — |

Further evidence that the paper does not report:

| Topic | Script | Output |
|---|---|---|
| One numerical setting at a time | `validation/paper1_onefactor.py onefactor` | `paper1_onefactor_onefactor.json` |
| Long-maturity decay rate | `validation/paper1_decay.py` | `paper1_decay.json` |
| Earlier versions of the benchmark: ADI against Crank–Nicolson, random grid placement, time-step refinement, the earlier cost tables | `validation/adi_vs_cn_bench.py`, `validation/paper1_fdm_shift.py`, `validation/paper1_accuracy_checks.py timestep conv_angular vega_fixed`, `validation/paper1_market_cost.py`, `figures/cost.py` | as named in each script |

The stored Monte Carlo runs of each contract, and the earlier bump-and-revalue
finite-difference Greeks, are in `validation/<contract>.toml`.

Each finite-difference solve at `m = 320` or `N = 640` takes up to a minute; on a
machine with little memory, run them one at a time. The Monte Carlo references take hours.

## Sealed holdouts

`configs/holdout_*.toml` were written down before evaluation and evaluated once, with
the outcome reported whatever it was. The records are
`validation/holdout_evaluation.json` (holdout A and a Bessel holdout, July 2026, at
settings that still pre-smoothed the payoff) and `validation/holdout_evaluation_r4.json`
(holdouts B and C, at the settings of the paper). Besides the six gates of the paper,
these records contain a seventh gate for a local-volatility correction that is not part
of this paper; its code is not included here. Digests and dates of the sealed files
are in `validation/HOLDOUT_SEALS.md`.

`python -m twoasset.holdout_eval` repeats the evaluation of holdouts B and C and writes
`validation/holdout_rerun.json`, leaving the records untouched.

## Tests

```bash
PYTHONPATH=. pytest tests
```

The tests check the Mathieu functions against their differential equations, the
pointwise and Galerkin behaviour of the three bases, the six production prices
against stored values, and the finite differences in whitened coordinates against a
closed form and against their stored grid levels. They take about four minutes.
