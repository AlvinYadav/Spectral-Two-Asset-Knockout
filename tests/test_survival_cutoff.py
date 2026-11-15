"""
Survival probability, exit-time density, hazard rate and rebate at hit from the
eigenvalue-cutoff bases, gated on validation/artifacts/paper1_survival.json
(written by validation/paper1_survival.py):

  I  S(tau) equals exp(r tau) times the unit-payoff price from the ordinary
     pricing path, to 1e-12 relative (three horizons, two spots, each contract);
  I  the density equals minus a central difference of S in tau, to 1e-8;
  S  S(T) and the rebate at hit lie within three standard errors of the
     bridge-corrected Monte Carlo hitting-time sampler;
  R  at the centre of the Euclidean basket, survival, density, hazard and digital
     price agree with an independent implementation to its six printed digits,
     and the long-run hazard rate kappa_1 to seven.

The rebate at hit is gated against Monte Carlo only: its by-parts form carries a
slowly convergent sum of A_n / omega_n (the survival function below tau_min), and
the artefact's `cutoff` section records how it moves with the cutoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._evidence import require_artefact
from twoasset.runlog import assert_gate

ROOT = Path(__file__).parent.parent
PATH = ROOT / "validation" / "artifacts" / "paper1_survival.json"
CURVED = ("default", "tight_barrier", "mahalanobis")


@pytest.fixture(scope="module")
def data(request):
    require_artefact(PATH, regen_command="python validation/paper1_survival.py",
                     request=request, cost_note="(about 5 minutes)")
    return json.loads(PATH.read_text())


@pytest.mark.parametrize("name", CURVED)
def test_survival_identity_and_density(data, name, runlog):
    d = data[name]
    assert_gate(runlog, "survival_vs_digital_rel", d["identity"]["max_rel_diff"], 1e-12,
                cmp="<", gate_class="I", config=name)
    assert_gate(runlog, "density_vs_central_diff_abs", d["ddtau"]["max_abs_diff"], 1e-8,
                cmp="<", gate_class="I", config=name, h=d["ddtau"]["h"])


@pytest.mark.parametrize("name", CURVED)
def test_survival_and_rebate_against_monte_carlo(data, name, runlog):
    mc = data[name]["mc"]
    for key in ("survival", "rebate"):
        assert_gate(runlog, f"{key}_mc_n_sem", mc[key]["n_sem"], 3.0, cmp="<=",
                    gate_class="S", config=name, spectral=mc[key]["spectral"],
                    mc=mc[key]["mc"], sem=mc[key]["sem"], n_paths=mc["n_paths"],
                    n_steps=mc["n_steps"], seed=mc["seed"])


def test_euclidean_regression_targets(data):
    d = data["default"]
    t = d["spec_targets"]
    assert d["kappa1"] == pytest.approx(t["kappa1"], abs=5e-8)
    for row in d["table"]:
        S, f, h, dig, _reb = t["rows"][str(row["tau"])]
        # The targets come from 100 x 192 nodes and a basis to lambda^2 = 164; the
        # production quadrature (50 x 96) differs from them by up to about 2e-5 of the
        # value at tau = T/4 (Table 6.7), hence 5e-6 absolute.
        for got, want in ((row["survival"], S), (row["density"], f),
                          (row["hazard"], h), (row["digital"], dig)):
            assert got == pytest.approx(want, abs=5e-6), (row["tau"], got, want)
