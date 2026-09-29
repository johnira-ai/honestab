# honestab

**A/B tests with honest error bars.**

[![tests](https://github.com/johnira-ai/honestab/actions/workflows/tests.yml/badge.svg)](https://github.com/johnira-ai/honestab/actions/workflows/tests.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

honestab is a small A/B testing toolkit where every statistical method comes
with a simulation-based calibration test. Before a method ships, it is run on
thousands of simulated experiments where the true effect is known, and it has
to prove that its p-values and confidence intervals mean what they claim.

## Why calibration matters

A 95% confidence interval is a promise: across many experiments, it should
contain the true effect 95% of the time. A test at `alpha = 0.05` makes a
similar promise: when there is no real effect, it should declare a winner only
5% of the time.

Plenty of A/B testing code quietly breaks these promises. A wrong variance
formula, a pooled standard error in the wrong place, or a test that assumes
equal variances when the arms differ can all look fine on a single dataset.
They are only visible in aggregate, and in practice that means shipping
features that did nothing, or missing ones that worked.

Calibration testing checks the promises directly:

1. Simulate thousands of experiments with a known true effect, including zero.
2. Analyze each one with the method under test.
3. Check that the false positive rate is close to `alpha`, that confidence
   intervals cover the true effect `1 - alpha` of the time, and that p-values
   are uniformly distributed under the null.

If a method can't pass that, its error bars aren't honest, and it doesn't ship.

## Installation

honestab uses [uv](https://docs.astral.sh/uv/) and requires Python 3.11+.

```bash
git clone https://github.com/johnira-ai/honestab.git
cd honestab
uv sync
```

## Usage

```python
from honestab import analyze, simulate

# Simulate an experiment: 10% baseline conversion, true lift of +1 point.
df = simulate(10_000, metric="binary", baseline=0.10, effect=0.01, seed=42)

result = analyze(df)
print(result)
# two-proportion z-test: effect = +0.0132 (95% CI [+0.0047, +0.0217]), p = 0.002339

result.relative_lift  # 0.134 -> +13.4% relative to control
result.significant    # True
```

`simulate()` returns a DataFrame with `group` and `value` columns, and records
the ground truth in `df.attrs["true_effect"]`. `analyze()` works on any
DataFrame in that shape, so you can pass it real experiment data too:

```python
result = analyze(my_df, metric="continuous", group_col="variant",
                 value_col="revenue", control="A", treatment="B")
```

Supported methods:

| Metric       | Test                                   | Confidence interval         |
| ------------ | -------------------------------------- | --------------------------- |
| `binary`     | Two-proportion z-test (pooled SE)      | Wald interval (unpooled SE) |
| `continuous` | Welch's t-test (unequal variances)     | Welch interval              |

## Running the tests

```bash
uv run ruff check
uv run pytest
```

The calibration suite in `tests/test_calibration.py` runs 2,000 simulated
experiments per check and verifies, for both binary and continuous metrics:

- the false positive rate under an A/A test matches `alpha`;
- confidence intervals cover the true effect at the nominal rate;
- p-values are uniform under the null (Kolmogorov–Smirnov test).

Tolerances are Monte Carlo bands of four binomial standard errors, and seeds
are fixed so results are reproducible.

## Roadmap

Each item will ship with its own calibration test.

- [ ] **SRM checks**: detect sample ratio mismatch before trusting results
- [ ] **CUPED**: variance reduction using pre-experiment covariates
- [ ] **Delta method**: correct standard errors for ratio metrics (e.g. revenue per session)
- [ ] **Sequential testing**: valid inference when peeking at results early
- [ ] **Power analysis**: sample size and minimum detectable effect calculations
- [ ] **Segment analysis**: per-segment effects with multiple comparison control
- [ ] **Streamlit dashboard**: interactive analysis and calibration reports

## License

[MIT](LICENSE)
