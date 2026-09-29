"""Analyze a two-arm A/B experiment.

``analyze`` estimates the absolute treatment effect (treatment minus control)
with a confidence interval and a two-sided p-value:

* binary metrics: two-proportion z-test (pooled variance under the null) with
  an unpooled Wald confidence interval;
* continuous metrics: Welch's t-test and Welch confidence interval, which do
  not assume equal variances.

Both are validated in ``tests/test_calibration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from honestab.simulate import CONTROL, TREATMENT


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
        level = round(100 * (1 - self.alpha))
        return (
            f"{self.method}: effect = {self.effect:+.4f} "
            f"({level}% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]), "
            f"p = {self.p_value:.4g}"
        )


def analyze(
    df: pd.DataFrame,
    *,
    metric: str | None = None,
    alpha: float = 0.05,
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
    group_col, value_col
        Column names for the arm label and the outcome.
    control, treatment
        Labels of the two arms in ``group_col``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    x_c = df.loc[df[group_col] == control, value_col].to_numpy(dtype=float)
    x_t = df.loc[df[group_col] == treatment, value_col].to_numpy(dtype=float)
    if len(x_c) < 2 or len(x_t) < 2:
        raise ValueError(
            f"need at least 2 units in each of {control!r} and {treatment!r}"
        )

    if metric is None:
        metric = df.attrs.get("metric")
    if metric is None:
        values = np.concatenate([x_c, x_t])
        metric = "binary" if np.isin(values, (0.0, 1.0)).all() else "continuous"

    if metric == "binary":
        return _two_proportion_z(x_c, x_t, alpha)
    if metric == "continuous":
        return _welch_t(x_c, x_t, alpha)
    raise ValueError(f"unknown metric {metric!r}; use 'binary' or 'continuous'")


def _two_proportion_z(x_c: np.ndarray, x_t: np.ndarray, alpha: float) -> AnalysisResult:
    n_c, n_t = len(x_c), len(x_t)
    p_c, p_t = x_c.mean(), x_t.mean()
    effect = p_t - p_c

    # Test: pooled standard error, which is correct under H0: p_c == p_t.
    p_pool = (x_c.sum() + x_t.sum()) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    if se_pool > 0:
        p_value = 2 * stats.norm.sf(abs(effect) / se_pool)
    else:
        p_value = 1.0

    # Interval: unpooled standard error, valid whether or not H0 holds.
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z = stats.norm.ppf(1 - alpha / 2)

    return AnalysisResult(
        metric="binary",
        method="two-proportion z-test",
        n_control=n_c,
        n_treatment=n_t,
        control_mean=float(p_c),
        treatment_mean=float(p_t),
        effect=float(effect),
        ci_low=float(effect - z * se),
        ci_high=float(effect + z * se),
        p_value=float(p_value),
        alpha=alpha,
    )


def _welch_t(x_c: np.ndarray, x_t: np.ndarray, alpha: float) -> AnalysisResult:
    n_c, n_t = len(x_c), len(x_t)
    m_c, m_t = x_c.mean(), x_t.mean()
    effect = m_t - m_c

    v_c = x_c.var(ddof=1) / n_c
    v_t = x_t.var(ddof=1) / n_t
    se = np.sqrt(v_c + v_t)

    if se > 0:
        # Welch–Satterthwaite degrees of freedom.
        dof = (v_c + v_t) ** 2 / (v_c**2 / (n_c - 1) + v_t**2 / (n_t - 1))
        t_crit = stats.t.ppf(1 - alpha / 2, dof)
        p_value = 2 * stats.t.sf(abs(effect) / se, dof)
    else:
        t_crit = 0.0
        p_value = 1.0 if effect == 0 else 0.0

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
