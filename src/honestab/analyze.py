"""Analyze a two-arm A/B experiment.

``analyze`` estimates the absolute treatment effect (treatment minus control)
with a confidence interval and a two-sided p-value:

* binary metrics: two-proportion z-test (pooled variance under the null) with
  a Newcombe hybrid score confidence interval;
* continuous metrics: Welch's t-test and Welch confidence interval, which do
  not assume equal variances.

Both are validated in ``tests/test_calibration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from honestab.simulate import CONTROL, TREATMENT, Metric

NanPolicy = Literal["raise", "omit"]


@dataclass(frozen=True)
class AnalysisResult:
    """Result of an A/B test analysis. ``effect`` is treatment minus control."""

    metric: str
    method: str
    n_control: int
    n_treatment: int
    control_mean: float
    treatment_mean: float
    effect: float
    ci_low: float
    ci_high: float
    p_value: float
    alpha: float

    @property
    def relative_lift(self) -> float:
        """Point estimate of the relative lift, ``effect / control_mean``."""
        if self.control_mean == 0:
            return float("nan")
        return self.effect / self.control_mean

    @property
    def significant(self) -> bool:
        """Whether the test rejects the null at level ``alpha``."""
        return self.p_value < self.alpha

    def __str__(self) -> str:
        level = round(100 * (1 - self.alpha), 6)
        return (
            f"{self.method}: effect = {self.effect:+.4f} "
            f"({level:g}% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]), "
            f"p = {self.p_value:.4g}"
        )


def analyze(
    df: pd.DataFrame,
    *,
    metric: Metric | None = None,
    alpha: float = 0.05,
    nan_policy: NanPolicy = "raise",
    group_col: str = "group",
    value_col: str = "value",
    control: str = CONTROL,
    treatment: str = TREATMENT,
) -> AnalysisResult:
    """Estimate the treatment effect in an A/B experiment.

    Parameters
    ----------
    df
        One row per unit, with a group column and an outcome column.
    metric
        ``"binary"`` or ``"continuous"``. If omitted, uses ``df.attrs["metric"]``
        when present (as set by :func:`honestab.simulate`), otherwise infers
        ``"binary"`` when every value is 0 or 1.
    alpha
        Significance level; the confidence interval has level ``1 - alpha``.
    nan_policy
        What to do with missing outcomes: ``"raise"`` (the default) raises a
        ``ValueError``; ``"omit"`` drops them before analysis.
    group_col, value_col
        Column names for the arm label and the outcome.
    control, treatment
        Labels of the two arms in ``group_col``.

    Raises
    ------
    ValueError
        If ``alpha`` or ``nan_policy`` is invalid, an arm has fewer than 2
        units, the outcomes contain infinite values (or missing values with
        ``nan_policy="raise"``), or a binary metric has values other than 0
        and 1.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if nan_policy not in ("raise", "omit"):
        raise ValueError(f"unknown nan_policy {nan_policy!r}; use 'raise' or 'omit'")

    x_c = _arm_values(df, group_col, value_col, control, nan_policy)
    x_t = _arm_values(df, group_col, value_col, treatment, nan_policy)
    if len(x_c) < 2 or len(x_t) < 2:
        raise ValueError(
            f"need at least 2 units in each of {control!r} and {treatment!r}"
        )

    is_binary = bool(np.isin(np.concatenate([x_c, x_t]), (0.0, 1.0)).all())
    if metric is None:
        metric = df.attrs.get("metric")
    if metric is None:
        metric = "binary" if is_binary else "continuous"

    if metric == "binary":
        if not is_binary:
            raise ValueError(
                "binary metric needs every value to be 0 or 1; "
                "pass metric='continuous' for other outcomes"
            )
        return _two_proportion_z(x_c, x_t, alpha)
    if metric == "continuous":
        return _welch_t(x_c, x_t, alpha)
    raise ValueError(f"unknown metric {metric!r}; use 'binary' or 'continuous'")


def _arm_values(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    label: str,
    nan_policy: NanPolicy,
) -> np.ndarray:
    """Outcomes for one arm as floats, with missing values handled."""
    x = df.loc[df[group_col] == label, value_col].to_numpy(dtype=float)
    missing = np.isnan(x)
    if missing.any():
        if nan_policy == "raise":
            raise ValueError(
                f"{missing.sum()} missing value(s) in arm {label!r}; "
                "drop them first or pass nan_policy='omit'"
            )
        x = x[~missing]
    if np.isinf(x).any():
        raise ValueError(f"infinite value(s) in arm {label!r}")
    return x


def _two_proportion_z(x_c: np.ndarray, x_t: np.ndarray, alpha: float) -> AnalysisResult:
    n_c, n_t = len(x_c), len(x_t)
    p_c, p_t = x_c.mean(), x_t.mean()
    effect = p_t - p_c

    # Test: pooled standard error, which is correct under H0: p_c == p_t.
    p_pool = (x_c.sum() + x_t.sum()) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    if se_pool == 0:
        p_value = 1.0
    else:
        p_value = 2 * stats.norm.sf(abs(effect) / se_pool)

    # Interval: Newcombe's hybrid score interval, which combines the Wilson
    # intervals of each arm. Unlike the Wald interval it keeps its coverage
    # with small samples and rates near 0 or 1.
    z = stats.norm.ppf(1 - alpha / 2)
    low_c, high_c = _wilson(p_c, n_c, z)
    low_t, high_t = _wilson(p_t, n_t, z)
    ci_low = effect - np.sqrt((p_t - low_t) ** 2 + (high_c - p_c) ** 2)
    ci_high = effect + np.sqrt((high_t - p_t) ** 2 + (p_c - low_c) ** 2)

    return AnalysisResult(
        metric="binary",
        method="two-proportion z-test",
        n_control=n_c,
        n_treatment=n_t,
        control_mean=float(p_c),
        treatment_mean=float(p_t),
        effect=float(effect),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        alpha=alpha,
    )


def _wilson(p: float, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return center - half, center + half


def _welch_t(x_c: np.ndarray, x_t: np.ndarray, alpha: float) -> AnalysisResult:
    n_c, n_t = len(x_c), len(x_t)
    m_c, m_t = x_c.mean(), x_t.mean()
    effect = m_t - m_c

    v_c = x_c.var(ddof=1) / n_c
    v_t = x_t.var(ddof=1) / n_t
    se = np.sqrt(v_c + v_t)

    if se == 0:
        # Both arms are constant, so the difference is known exactly.
        t_crit = 0.0
        p_value = 1.0 if effect == 0 else 0.0
    else:
        # Welch–Satterthwaite degrees of freedom.
        dof = (v_c + v_t) ** 2 / (v_c**2 / (n_c - 1) + v_t**2 / (n_t - 1))
        t_crit = stats.t.ppf(1 - alpha / 2, dof)
        p_value = 2 * stats.t.sf(abs(effect) / se, dof)

    return AnalysisResult(
        metric="continuous",
        method="Welch's t-test",
        n_control=n_c,
        n_treatment=n_t,
        control_mean=float(m_c),
        treatment_mean=float(m_t),
        effect=float(effect),
        ci_low=float(effect - t_crit * se),
        ci_high=float(effect + t_crit * se),
        p_value=float(p_value),
        alpha=alpha,
    )
