"""Unit tests for ``srm_check``: hand-computed values and input handling."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from honestab import srm_check


def make_df(n_control: int, n_treatment: int, n_other: int = 0) -> pd.DataFrame:
    labels = ["control", "treatment", "holdout"]
    return pd.DataFrame({"group": np.repeat(labels, [n_control, n_treatment, n_other])})


def test_matches_hand_computed_chi_square() -> None:
    # Designed 50/50; each arm expects 50,600 and is off by 600.
    result = srm_check(make_df(50_000, 51_200))
    chi2 = 2 * 600**2 / 50_600
    assert result.chi2 == pytest.approx(chi2)
    # With one degree of freedom, chi-square is a squared standard normal.
    assert result.p_value == pytest.approx(2 * stats.norm.sf(np.sqrt(chi2)))
    assert result.mismatch
    assert result.observed_share == pytest.approx(51_200 / 101_200)


def test_uneven_designed_split() -> None:
    # Designed 90/10 on 10,000 units: expected 9,000 and 1,000, both off by 120.
    result = srm_check(make_df(9_120, 880), expected_share=0.1)
    assert result.chi2 == pytest.approx(120**2 / 9_000 + 120**2 / 1_000)
    assert result.mismatch


def test_exact_split_has_p_value_one() -> None:
    result = srm_check(make_df(5_000, 5_000))
    assert result.p_value == pytest.approx(1.0)
    assert not result.mismatch
    assert "no mismatch" in str(result)


def test_alpha_sets_the_threshold() -> None:
    df = make_df(5_000, 5_250)  # chi2 = 250**2 / 10_250, so p is about 0.014
    assert not srm_check(df).mismatch
    assert srm_check(df, alpha=0.05).mismatch


def test_other_labels_are_ignored() -> None:
    result = srm_check(make_df(5_000, 5_000, n_other=3_000))
    assert (result.n_control, result.n_treatment) == (5_000, 5_000)


def test_custom_column_and_labels() -> None:
    df = pd.DataFrame({"variant": ["A"] * 40 + ["B"] * 60})
    result = srm_check(df, group_col="variant", control="A", treatment="B")
    assert (result.n_control, result.n_treatment) == (40, 60)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"expected_share": 0.0}, "expected_share"),
        ({"expected_share": 1.0}, "expected_share"),
        ({"alpha": 0.0}, "alpha"),
        ({"control": "A", "treatment": "B"}, "no units"),
    ],
)
def test_invalid_arguments_raise(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        srm_check(make_df(100, 100), **kwargs)  # type: ignore[arg-type]
