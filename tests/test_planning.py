"""Unit tests for power analysis: reference values and input handling."""

from collections.abc import Callable

import numpy as np
import pytest
from scipy import stats

from honestab import minimum_detectable_effect, power, sample_size
from honestab.simulate import Metric

BASELINE = 0.10
SD = 20.0


def fleiss_sample_size(p_c: float, p_t: float, alpha: float, target: float) -> int:
    """Textbook closed form for the pooled two-proportion z-test (no correction)."""
    z_alpha, z_beta = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(target)
    p_bar = (p_c + p_t) / 2
    numerator = z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(
        p_c * (1 - p_c) + p_t * (1 - p_t)
    )
    return int(np.ceil(numerator**2 / (p_t - p_c) ** 2))


@pytest.mark.parametrize(
    ("baseline", "effect", "alpha", "target"),
    [
        (0.10, 0.02, 0.05, 0.80),
        (0.05, 0.01, 0.05, 0.80),
        (0.30, -0.03, 0.05, 0.80),
        (0.10, 0.02, 0.01, 0.90),
    ],
)
def test_binary_sample_size_matches_closed_form(
    baseline: float, effect: float, alpha: float, target: float
) -> None:
    n = sample_size(effect, baseline=baseline, alpha=alpha, power=target)
    assert n == fleiss_sample_size(baseline, baseline + effect, alpha, target)


@pytest.mark.parametrize(
    ("n", "effect", "sd", "expected"),
    [
        # Reference values from statsmodels' TTestIndPower().power(effect / sd, n).
        (50, 0.5, 2.0, 0.23577967319947096),
        (400, 0.5, 2.0, 0.941944901182658),
        (400, 1.5, 20.0, 0.18517244104430594),
    ],
)
def test_continuous_power_matches_reference(
    n: int, effect: float, sd: float, expected: float
) -> None:
    assert power(n, effect, metric="continuous", sd=sd) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("effect", "sd", "expected"),
    [
        # ceil(TTestIndPower().solve_power(effect / sd, power=0.8, alpha=0.05)).
        (0.5, 2.0, 253),
        (1.5, 20.0, 2_792),
    ],
)
def test_continuous_sample_size_matches_reference(
    effect: float, sd: float, expected: int
) -> None:
    assert sample_size(effect, metric="continuous", sd=sd) == expected


@pytest.mark.parametrize(
    ("metric", "effect"),
    [("binary", 0.02), ("binary", -0.02), ("continuous", 1.5)],
)
def test_sample_size_is_the_smallest_that_reaches_the_target(
    metric: Metric, effect: float
) -> None:
    n = sample_size(effect, metric=metric, baseline=BASELINE, sd=SD)
    assert power(n, effect, metric=metric, baseline=BASELINE, sd=SD) >= 0.80
    assert power(n - 1, effect, metric=metric, baseline=BASELINE, sd=SD) < 0.80


@pytest.mark.parametrize(
    ("metric", "n"),
    [("binary", 3_000), ("binary", 50), ("continuous", 500)],
)
def test_minimum_detectable_effect_round_trips(metric: Metric, n: int) -> None:
    mde = minimum_detectable_effect(n, metric=metric, baseline=BASELINE, sd=SD)
    assert mde > 0
    achieved = power(n, mde, metric=metric, baseline=BASELINE, sd=SD)
    assert achieved == pytest.approx(0.80)


def test_power_under_the_null_is_alpha() -> None:
    assert power(1_000, 0.0, alpha=0.05) == pytest.approx(0.05)
    assert power(1_000, 0.0, metric="continuous", alpha=0.01) == pytest.approx(0.01)


def test_continuous_power_is_symmetric_and_finite_for_huge_effects() -> None:
    up = power(500, 3.0, metric="continuous", sd=SD)
    down = power(500, -3.0, metric="continuous", sd=SD)
    assert up == pytest.approx(down)
    # Deep in the tail, scipy's noncentral t CDF can return NaN.
    assert power(500, 200.0, metric="continuous", sd=SD) == 1.0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: sample_size(0.0), "nonzero"),
        (lambda: sample_size(0.02, power=0.01), "power"),
        (lambda: sample_size(0.02, alpha=1.0), "alpha"),
        (lambda: sample_size(0.95, baseline=0.10), r"baseline \+ effect"),
        (lambda: sample_size(0.02, baseline=0.0), "baseline"),
        (lambda: sample_size(1.0, metric="continuous", sd=0.0), "sd"),
        (lambda: sample_size(0.02, metric="ratio"), "unknown metric"),  # type: ignore[arg-type]
        (lambda: power(1, 0.02), "at least 2"),
        (lambda: minimum_detectable_effect(10, baseline=0.99), "no effect"),
    ],
)
def test_invalid_arguments_raise(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
