# honestab

honestab is an A/B testing toolkit whose selling point is that every statistical
method is validated with simulation-based calibration tests: we simulate
experiments with a known ground truth and check that p-values, confidence
intervals, and error rates behave the way the theory promises.

## Structure

```
src/honestab/
  __init__.py     # public API
  simulate.py     # simulate A/B experiments with known true effects
  analyze.py      # statistical analysis of experiment data
tests/
  test_calibration.py  # simulation-based calibration tests
.github/workflows/tests.yml  # CI: uv sync, ruff check, pytest
```

## Tooling

- Package and environment management: `uv` (Python 3.11+)
- Install: `uv sync`
- Lint: `uv run ruff check` (config in `pyproject.toml`, line length 88)
- Test: `uv run pytest`

## The rule

**Every new statistical method must ship with a calibration test that checks it
against simulated ground truth.** A method without a calibration test does not
get merged. At minimum, the test should simulate many experiments under a known
effect (including the null) and verify that the method's error rates and
interval coverage match their nominal levels within Monte Carlo tolerance.
