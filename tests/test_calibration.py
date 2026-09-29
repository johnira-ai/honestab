"""Simulation-based calibration tests.

Each test simulates many experiments with a known true effect and checks that
the analysis behaves as advertised: tests reject a true null at rate ``alpha``,
confidence intervals cover the true effect at rate ``1 - alpha``, and p-values
are uniform under the null.

Tolerances are Monte Carlo bands: an observed rate must fall within
``Z_TOL`` binomial standard errors of its nominal value. Seeds are fixed, so
results are reproducible.
"""

import numpy as np
import pytest
from scipy import stats

from honestab import analyze, simulate

N_SIMS = 2_000
N_PER_ARM = 2_000
ALPHA = 0.05
Z_TOL = 4.0  # a correct method falls outside this band ~6 times in 100,000


def run_experiments(seed: int, **sim_kwargs):
    """Simulate and analyze N_SIMS independent experiments."""
    rngs = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(N_SIMS)
    ]
    return [
        analyze(simulate(N_PER_ARM, seed=rng, **sim_kwargs), alpha=ALPHA)
        for rng in rngs
    ]


def assert_rate_close(observed: float, nominal: float, what: str) -> None:
    se = np.sqrt(nominal * (1 - nominal) / N_SIMS)
    assert abs(observed - nominal) <= Z_TOL * se, (
        f"{what}: observed {observed:.4f}, nominal {nominal:.4f}, "
        f"allowed ±{Z_TOL * se:.4f}"
    )


@pytest.mark.parametrize(
    ("metric", "baseline"),
    [("binary", 0.10), ("continuous", 50.0)],
)
def test_false_positive_rate_matches_alpha(metric, baseline):
    """Under an A/A test, the test should reject about ALPHA of the time."""
    results = run_experiments(seed=1, metric=metric, baseline=baseline, sd=20.0)
    rejection_rate = np.mean([r.p_value < ALPHA for r in results])
    assert_rate_close(rejection_rate, ALPHA, f"{metric} type I error")


@pytest.mark.parametrize(
    ("metric", "baseline", "effect"),
    [("binary", 0.10, 0.01), ("continuous", 50.0, 1.5)],
)
def test_confidence_interval_coverage(metric, baseline, effect):
    """With a real effect, the CI should contain it about 1 - ALPHA of the time."""
    results = run_experiments(
        seed=2, metric=metric, baseline=baseline, effect=effect, sd=20.0
    )
    coverage = np.mean([r.ci_low <= effect <= r.ci_high for r in results])
    assert_rate_close(coverage, 1 - ALPHA, f"{metric} CI coverage")


def test_p_values_uniform_under_null():
    """Under the null, the whole p-value distribution should be Uniform(0, 1)."""
    results = run_experiments(seed=3, metric="continuous", baseline=50.0, sd=20.0)
    p_values = np.array([r.p_value for r in results])
    ks = stats.kstest(p_values, "uniform")
    assert ks.pvalue > 0.001, f"p-values not uniform (KS p = {ks.pvalue:.2e})"
