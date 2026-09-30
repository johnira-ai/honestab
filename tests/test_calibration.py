"""Simulation-based calibration tests.

Each test simulates many experiments with a known true effect and checks that
the analysis behaves as advertised: tests reject a true null at rate ``alpha``,
confidence intervals cover the true effect at rate ``1 - alpha``, p-values
are uniform under the null, SRM checks raise false alarms at rate ``alpha``,
and experiments sized with the power functions detect effects as often as
promised.

Tolerances are Monte Carlo bands: an observed rate must fall within
``Z_TOL`` binomial standard errors of its nominal value. Seeds are fixed, so
results are reproducible.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from honestab import (
    AnalysisResult,
    SRMResult,
    analyze,
    power,
    sample_size,
    simulate,
    srm_check,
)
from honestab.simulate import Metric

N_SIMS = 2_000
N_PER_ARM = 2_000
ALPHA = 0.05
Z_TOL = 4.0  # a correct method falls outside this band ~6 times in 100,000


def run_experiments(
    seed: int,
    n_per_arm: int = N_PER_ARM,
    n_sims: int = N_SIMS,
    **sim_kwargs: Any,
) -> list[AnalysisResult]:
    """Simulate and analyze ``n_sims`` independent experiments."""
    rngs = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_sims)
    ]
    return [
        analyze(simulate(n_per_arm, seed=rng, **sim_kwargs), alpha=ALPHA)
        for rng in rngs
    ]


def monte_carlo_tolerance(nominal: float, n_sims: int) -> float:
    return float(Z_TOL * np.sqrt(nominal * (1 - nominal) / n_sims))


def assert_rate_close(
    observed: float, nominal: float, what: str, n_sims: int = N_SIMS
) -> None:
    tol = monte_carlo_tolerance(nominal, n_sims)
    assert abs(observed - nominal) <= tol, (
        f"{what}: observed {observed:.4f}, nominal {nominal:.4f}, allowed ±{tol:.4f}"
    )


def assert_rate_at_least(
    observed: float, nominal: float, what: str, n_sims: int = N_SIMS
) -> None:
    tol = monte_carlo_tolerance(nominal, n_sims)
    assert observed >= nominal - tol, (
        f"{what}: observed {observed:.4f}, nominal {nominal:.4f}, "
        f"allowed down to {nominal - tol:.4f}"
    )


@pytest.mark.parametrize(
    ("metric", "baseline"),
    [("binary", 0.10), ("continuous", 50.0)],
)
def test_false_positive_rate_matches_alpha(metric: str, baseline: float) -> None:
    """Under an A/A test, the test should reject about ALPHA of the time."""
    results = run_experiments(seed=1, metric=metric, baseline=baseline, sd=20.0)
    rejection_rate = float(np.mean([r.p_value < ALPHA for r in results]))
    assert_rate_close(rejection_rate, ALPHA, f"{metric} type I error")


@pytest.mark.parametrize(
    ("metric", "baseline", "effect"),
    [("binary", 0.10, 0.01), ("continuous", 50.0, 1.5)],
)
def test_confidence_interval_coverage(
    metric: str, baseline: float, effect: float
) -> None:
    """With a real effect, the CI should contain it about 1 - ALPHA of the time."""
    results = run_experiments(
        seed=2, metric=metric, baseline=baseline, effect=effect, sd=20.0
    )
    coverage = float(np.mean([r.ci_low <= effect <= r.ci_high for r in results]))
    assert_rate_close(coverage, 1 - ALPHA, f"{metric} CI coverage")


def test_binary_confidence_interval_coverage_small_sample() -> None:
    """With few units and a rare event, the binary CI must not under-cover.

    Outcomes are discrete here, so no interval can hit 1 - ALPHA exactly; an
    honest one errs on the side of covering more often. The Wald interval
    fails this check (coverage about 0.93).
    """
    n_sims = 5_000
    effect = 0.01
    results = run_experiments(
        seed=4,
        n_per_arm=100,
        n_sims=n_sims,
        metric="binary",
        baseline=0.02,
        effect=effect,
    )
    coverage = float(np.mean([r.ci_low <= effect <= r.ci_high for r in results]))
    assert_rate_at_least(
        coverage, 1 - ALPHA, "small-sample binary CI coverage", n_sims=n_sims
    )


@pytest.mark.parametrize(
    ("metric", "baseline"),
    [("binary", 0.10), ("continuous", 50.0)],
)
def test_p_values_uniform_under_null(metric: str, baseline: float) -> None:
    """Under the null, the whole p-value distribution should be Uniform(0, 1)."""
    results = run_experiments(seed=3, metric=metric, baseline=baseline, sd=20.0)
    p_values = np.array([r.p_value for r in results])
    ks = stats.kstest(p_values, "uniform")
    assert ks.pvalue > 0.001, f"p-values not uniform (KS p = {ks.pvalue:.2e})"


# Power analysis ------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "baseline", "effect", "target"),
    [
        ("binary", 0.10, 0.02, 0.80),
        ("binary", 0.10, 0.02, 0.50),
        ("continuous", 50.0, 1.5, 0.80),
    ],
)
def test_power_matches_simulated_detection_rate(
    metric: Metric, baseline: float, effect: float, target: float
) -> None:
    """An experiment sized for a target power should detect the effect that often.

    The rejection rate is what ``analyze`` actually achieves on simulated
    experiments at the sample size ``sample_size`` recommends, so this checks
    the planning formulas against the real tests, not against themselves.
    """
    sd = 20.0
    n = sample_size(
        effect, metric=metric, baseline=baseline, sd=sd, alpha=ALPHA, power=target
    )
    promised = power(n, effect, metric=metric, baseline=baseline, sd=sd, alpha=ALPHA)

    results = run_experiments(
        seed=5, n_per_arm=n, metric=metric, baseline=baseline, effect=effect, sd=sd
    )
    detection_rate = float(np.mean([r.p_value < ALPHA for r in results]))
    assert_rate_close(detection_rate, promised, f"{metric} power at n={n}")


# Sample ratio mismatch -----------------------------------------------------


def run_srm_checks(
    seed: int, n_units: int, true_share: float, expected_share: float, alpha: float
) -> list[SRMResult]:
    """Randomly assign ``n_units`` per experiment and run the SRM check."""
    rng = np.random.default_rng(seed)
    n_treatment = rng.binomial(n_units, true_share, size=N_SIMS)
    return [
        srm_check(
            pd.DataFrame(
                {"group": np.repeat(["control", "treatment"], [n_units - k, k])}
            ),
            expected_share=expected_share,
            alpha=alpha,
        )
        for k in n_treatment
    ]


@pytest.mark.parametrize("share", [0.5, 0.1])
def test_srm_false_alarm_rate_matches_alpha(share: float) -> None:
    """With correct randomization, the SRM check should fire ALPHA of the time."""
    results = run_srm_checks(
        seed=6, n_units=10_000, true_share=share, expected_share=share, alpha=ALPHA
    )
    false_alarm_rate = float(np.mean([r.mismatch for r in results]))
    assert_rate_close(false_alarm_rate, ALPHA, f"SRM false alarms at {share:.0%}")


def test_srm_p_values_uniform_under_null() -> None:
    results = run_srm_checks(
        seed=7, n_units=10_000, true_share=0.5, expected_share=0.5, alpha=ALPHA
    )
    ks = stats.kstest([r.p_value for r in results], "uniform")
    assert ks.pvalue > 0.001, f"SRM p-values not uniform (KS p = {ks.pvalue:.2e})"


def test_srm_detection_rate_matches_theory() -> None:
    """A real mismatch should be caught as often as the chi-square test predicts.

    Under a true treatment share ``s1`` against a designed share ``s0``, the
    statistic is approximately noncentral chi-square with one degree of
    freedom and noncentrality ``n (s1 - s0)^2 / (s0 (1 - s0))``.
    """
    n_units, designed, actual, alpha = 20_000, 0.5, 0.51, 0.001
    results = run_srm_checks(
        seed=8,
        n_units=n_units,
        true_share=actual,
        expected_share=designed,
        alpha=alpha,
    )
    detection_rate = float(np.mean([r.mismatch for r in results]))
    noncentrality = n_units * (actual - designed) ** 2 / (designed * (1 - designed))
    predicted = float(stats.ncx2.sf(stats.chi2.isf(alpha, 1), 1, noncentrality))
    assert_rate_close(detection_rate, predicted, "SRM detection rate")
