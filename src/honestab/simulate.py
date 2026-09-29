"""Simulate A/B experiments with a known ground-truth effect.

Every statistical method in honestab is checked against data from this module:
because we choose the true effect, we can measure how often a method's
confidence intervals cover it and how often its tests reject when they
should not.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Metric = Literal["binary", "continuous"]

CONTROL = "control"
TREATMENT = "treatment"


def simulate(
    n_control: int = 5_000,
    n_treatment: int | None = None,
    *,
    metric: Metric = "binary",
    baseline: float = 0.10,
    effect: float = 0.0,
    sd: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate one two-arm A/B experiment.

    Parameters
    ----------
    n_control, n_treatment
        Number of units in each arm. ``n_treatment`` defaults to ``n_control``.
    metric
        ``"binary"`` draws Bernoulli outcomes (e.g. conversion); ``"continuous"``
        draws normal outcomes (e.g. revenue per user).
    baseline
        Control-arm conversion rate (binary) or mean (continuous).
    effect
        True absolute treatment effect: treatment minus control, on the metric
        scale. ``0.0`` simulates an A/A test (the null hypothesis is true).
    sd
        Standard deviation of each arm, used only for continuous metrics.
    seed
        Seed or ``numpy.random.Generator`` for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per unit with columns ``group`` (``"control"`` or
        ``"treatment"``) and ``value``. The ground truth is stored in
        ``df.attrs["true_effect"]`` and ``df.attrs["metric"]``.
    """
    if n_treatment is None:
        n_treatment = n_control
    if n_control < 2 or n_treatment < 2:
        raise ValueError("each arm needs at least 2 units")

    rng = np.random.default_rng(seed)

    if metric == "binary":
        p_treatment = baseline + effect
        if not (0.0 <= baseline <= 1.0 and 0.0 <= p_treatment <= 1.0):
            raise ValueError(
                "binary metric needs baseline and baseline + effect in [0, 1]"
            )
        control = rng.binomial(1, baseline, size=n_control).astype(float)
        treatment = rng.binomial(1, p_treatment, size=n_treatment).astype(float)
    elif metric == "continuous":
        if sd <= 0:
            raise ValueError("sd must be positive")
        control = rng.normal(baseline, sd, size=n_control)
        treatment = rng.normal(baseline + effect, sd, size=n_treatment)
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'binary' or 'continuous'")

    df = pd.DataFrame(
        {
            "group": np.repeat([CONTROL, TREATMENT], [n_control, n_treatment]),
            "value": np.concatenate([control, treatment]),
        }
    )
    df.attrs["true_effect"] = float(effect)
    df.attrs["metric"] = metric
    return df
