"""Plan an experiment: power, sample size and minimum detectable effect.

The formulas model the tests that :func:`honestab.analyze` runs, so the power
promised here is the power ``analyze`` delivers:

* binary metrics: the two-proportion z-test with a pooled standard error under
  the null and an unpooled one under the alternative (normal approximation);
* continuous metrics: the two-sample t-test with equal arm sizes and a common
  standard deviation, using the noncentral t distribution. With equal arms and
  variances, Welch's degrees of freedom are ``2n - 2``.

All three functions assume equal arm sizes and a two-sided test, and are
validated in ``tests/test_calibration.py``.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from honestab.simulate import Metric


def power(
    n_per_arm: int,
    effect: float,
    *,
    metric: Metric = "binary",
    baseline: float = 0.10,
    sd: float = 1.0,
    alpha: float = 0.05,
) -> float:
    """Probability that ``analyze`` detects a true ``effect`` at level ``alpha``.

    Parameters
    ----------
    n_per_arm
        Number of units in each arm.
    effect
        True absolute treatment effect (treatment minus control).
    metric
        ``"binary"`` or ``"continuous"``.
    baseline
        Control-arm conversion rate (binary only).
    sd
        Standard deviation of each arm (continuous only).
    alpha
        Significance level of the two-sided test.
    """
    _validate(metric, baseline, sd, alpha)
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2")
    return _power(n_per_arm, effect, metric, baseline, sd, alpha)


def sample_size(
    effect: float,
    *,
    metric: Metric = "binary",
    baseline: float = 0.10,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Smallest number of units per arm that detects ``effect`` with ``power``.

    Parameters are as in :func:`power`; ``power`` is the target probability of
    detecting the effect.
    """
    _validate(metric, baseline, sd, alpha)
    _validate_target(power, alpha)
    if effect == 0:
        raise ValueError("effect must be nonzero")

    def achieved(n: int) -> float:
        return _power(n, effect, metric, baseline, sd, alpha)

    # Power increases with n: double until the target is met, then bisect.
    low, high = 2, 4
    while achieved(high) < power:
        low, high = high, high * 2
        if high > 10**12:
            raise ValueError("effect is too small to detect at any sample size")
    if achieved(low) >= power:
        return low
    while high - low > 1:
        mid = (low + high) // 2
        if achieved(mid) >= power:
            high = mid
        else:
            low = mid
    return high


def minimum_detectable_effect(
    n_per_arm: int,
    *,
    metric: Metric = "binary",
    baseline: float = 0.10,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest positive effect that ``n_per_arm`` units detect with ``power``.

    Parameters are as in :func:`power`. For binary metrics the effect is an
    increase from ``baseline``; decreases are not always symmetric.
    """
    _validate(metric, baseline, sd, alpha)
    _validate_target(power, alpha)
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2")

    def shortfall(effect: float) -> float:
        return _power(n_per_arm, effect, metric, baseline, sd, alpha) - power

    if metric == "binary":
        high = 1 - baseline
        if shortfall(high) < 0:
            raise ValueError(
                f"no effect up to a rate of 1 reaches power {power} "
                f"with {n_per_arm} units per arm"
            )
    else:
        high = sd
        while shortfall(high) < 0:
            high *= 2
    return float(optimize.brentq(shortfall, high * 1e-9, high, xtol=1e-12))


def _power(
    n: int, effect: float, metric: Metric, baseline: float, sd: float, alpha: float
) -> float:
    if metric == "binary":
        p_c, p_t = baseline, baseline + effect
        if not 0 <= p_t <= 1:
            raise ValueError("binary metric needs baseline + effect in [0, 1]")
        z = stats.norm.ppf(1 - alpha / 2)
        p_bar = (p_c + p_t) / 2
        se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n)
        se_alt = np.sqrt((p_c * (1 - p_c) + p_t * (1 - p_t)) / n)
        if se_alt == 0:
            return 1.0 if effect != 0 else 0.0
        upper = stats.norm.sf((z * se_null - effect) / se_alt)
        lower = stats.norm.cdf((-z * se_null - effect) / se_alt)
        return float(upper + lower)

    dof = 2 * n - 2
    noncentrality = effect / (sd * np.sqrt(2 / n))
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    upper = stats.nct.sf(t_crit, dof, noncentrality)
    # P(T < -t_crit) written as the mirrored upper tail: scipy's nct.cdf
    # returns NaN deep in the lower tail, while nct.sf stays accurate there.
    lower = stats.nct.sf(t_crit, dof, -noncentrality)
    return float(upper + lower)


def _validate(metric: str, baseline: float, sd: float, alpha: float) -> None:
    if metric not in ("binary", "continuous"):
        raise ValueError(f"unknown metric {metric!r}; use 'binary' or 'continuous'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if metric == "binary" and not 0 < baseline < 1:
        raise ValueError("binary metric needs baseline in (0, 1)")
    if metric == "continuous" and sd <= 0:
        raise ValueError("sd must be positive")


def _validate_target(power: float, alpha: float) -> None:
    if not alpha < power < 1:
        raise ValueError("power must be between alpha and 1")
