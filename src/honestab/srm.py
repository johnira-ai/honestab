"""Detect sample ratio mismatch (SRM).

If an experiment was designed to split users 50/50 but the arms end up with,
say, 50,000 and 51,200 users, something in the assignment or logging pipeline
is broken, and the effect estimate cannot be trusted no matter how small its
p-value. ``srm_check`` tests the observed arm sizes against the designed split
with a chi-square goodness-of-fit test.

The default threshold is ``alpha = 0.001`` rather than 0.05: an SRM check runs
on every experiment, so false alarms need to be rare, and real mismatches in
large experiments produce p-values far below it.

Validated in ``tests/test_calibration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats

from honestab.simulate import CONTROL, TREATMENT


@dataclass(frozen=True)
class SRMResult:
    """Result of a sample ratio mismatch check."""

    n_control: int
    n_treatment: int
    expected_share: float
    chi2: float
    p_value: float
    alpha: float

    @property
    def observed_share(self) -> float:
        """Share of units in the treatment arm."""
        return self.n_treatment / (self.n_control + self.n_treatment)

    @property
    def mismatch(self) -> bool:
        """Whether the arm sizes differ from the designed split at ``alpha``."""
        return self.p_value < self.alpha

    def __str__(self) -> str:
        verdict = "sample ratio mismatch" if self.mismatch else "no mismatch"
        return (
            f"SRM check: {verdict} (treatment share {self.observed_share:.4f}, "
            f"expected {self.expected_share:.4f}, p = {self.p_value:.4g})"
        )


def srm_check(
    df: pd.DataFrame,
    *,
    expected_share: float = 0.5,
    alpha: float = 0.001,
    group_col: str = "group",
    control: str = CONTROL,
    treatment: str = TREATMENT,
) -> SRMResult:
    """Test whether the arm sizes match the designed traffic split.

    Parameters
    ----------
    df
        One row per unit, with a group column. Other columns are ignored.
    expected_share
        Designed share of units in the treatment arm, e.g. ``0.5`` for a 50/50
        split or ``0.1`` for a 90/10 split.
    alpha
        Threshold for flagging a mismatch.
    group_col
        Column name for the arm label.
    control, treatment
        Labels of the two arms in ``group_col``. Rows with other labels are
        ignored.

    Raises
    ------
    ValueError
        If ``expected_share`` or ``alpha`` is not in (0, 1), or neither arm
        has any units.
    """
    if not 0 < expected_share < 1:
        raise ValueError("expected_share must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    groups = df[group_col]
    n_c = int((groups == control).sum())
    n_t = int((groups == treatment).sum())
    total = n_c + n_t
    if total == 0:
        raise ValueError(f"no units labeled {control!r} or {treatment!r}")

    expected = [total * (1 - expected_share), total * expected_share]
    chi2, p_value = stats.chisquare([n_c, n_t], f_exp=expected)

    return SRMResult(
        n_control=n_c,
        n_treatment=n_t,
        expected_share=expected_share,
        chi2=float(chi2),
        p_value=float(p_value),
        alpha=alpha,
    )
