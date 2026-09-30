"""Unit tests for ``analyze``: reference values and input handling.

The calibration suite checks that the methods behave correctly on average;
these tests pin exact numbers against independent implementations and check
that bad input fails loudly instead of producing a misleading result.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from honestab import analyze, simulate


def make_df(control: np.ndarray, treatment: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["control"] * len(control) + ["treatment"] * len(treatment),
            "value": np.concatenate([control, treatment]),
        }
    )


def binary_arm(successes: int, n: int) -> np.ndarray:
    return np.concatenate([np.ones(successes), np.zeros(n - successes)])


def test_welch_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    control, treatment = rng.normal(0, 1, 30), rng.normal(0.5, 3, 80)
    result = analyze(make_df(control, treatment), metric="continuous")
    reference = stats.ttest_ind(treatment, control, equal_var=False)
    ci = reference.confidence_interval()
    assert result.p_value == pytest.approx(reference.pvalue)
    assert result.ci_low == pytest.approx(ci.low)
    assert result.ci_high == pytest.approx(ci.high)


def test_two_proportion_z_matches_chi_square() -> None:
    # The pooled two-proportion z-test is equivalent to the uncorrected
    # chi-square test on the 2x2 table.
    result = analyze(make_df(binary_arm(10, 100), binary_arm(18, 100)))
    table = [[10, 90], [18, 82]]
    reference = stats.chi2_contingency(table, correction=False)
    assert result.p_value == pytest.approx(reference.pvalue)


@pytest.mark.parametrize(
    ("control", "treatment", "expected"),
    [
        # Reference values from statsmodels'
        # confint_proportions_2indep(method="newcomb", compare="diff").
        ((10, 100), (18, 100), (-0.017462646626674114, 0.17755378161808788)),
        ((0, 50), (3, 60), (-0.02854702483316282, 0.1370051647634852)),
    ],
)
def test_binary_ci_matches_newcombe_reference(
    control: tuple[int, int],
    treatment: tuple[int, int],
    expected: tuple[float, float],
) -> None:
    result = analyze(make_df(binary_arm(*control), binary_arm(*treatment)))
    assert (result.ci_low, result.ci_high) == pytest.approx(expected)


def test_missing_values_raise_by_default() -> None:
    df = simulate(100, metric="continuous", seed=0)
    df.loc[0, "value"] = np.nan
    with pytest.raises(ValueError, match="missing value"):
        analyze(df)


def test_nan_policy_omit_drops_missing_values() -> None:
    df = simulate(100, metric="continuous", seed=0)
    with_nan = df.copy()
    with_nan.loc[[0, 150], "value"] = np.nan
    result = analyze(with_nan, nan_policy="omit")
    expected = analyze(with_nan.dropna())
    assert result == expected
    assert (result.n_control, result.n_treatment) == (99, 99)


def test_infinite_values_raise() -> None:
    df = simulate(100, metric="continuous", seed=0)
    df.loc[0, "value"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        analyze(df)


def test_binary_metric_rejects_non_binary_values() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        analyze(
            make_df(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])), metric="binary"
        )


def test_stale_binary_attrs_do_not_hide_non_binary_values() -> None:
    # pandas keeps df.attrs through many operations, so a rescaled simulated
    # frame still claims to be binary.
    df = simulate(100, metric="binary", seed=0)
    df["value"] *= 100
    assert df.attrs["metric"] == "binary"
    with pytest.raises(ValueError, match="0 or 1"):
        analyze(df)


def test_metric_is_inferred_from_values() -> None:
    df = make_df(binary_arm(3, 10), binary_arm(5, 10))
    assert analyze(df).metric == "binary"
    df["value"] += 0.5
    assert analyze(df).metric == "continuous"


def test_constant_arms() -> None:
    same = analyze(make_df(np.ones(5), np.ones(5)), metric="continuous")
    assert same.p_value == 1.0
    different = analyze(make_df(np.ones(5), np.full(5, 2.0)), metric="continuous")
    assert different.p_value == 0.0
    assert different.ci_low == different.ci_high == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"nan_policy": "propagate"}, "nan_policy"),
        ({"metric": "ratio"}, "unknown metric"),
        ({"control": "A"}, "at least 2 units"),
    ],
)
def test_invalid_arguments_raise(kwargs: dict[str, object], message: str) -> None:
    df = simulate(100, metric="continuous", seed=0)
    with pytest.raises(ValueError, match=message):
        analyze(df, **kwargs)  # type: ignore[arg-type]


def test_str_shows_confidence_level() -> None:
    df = simulate(100, metric="continuous", seed=0)
    assert "95% CI" in str(analyze(df))
    assert "97.5% CI" in str(analyze(df, alpha=0.025))
