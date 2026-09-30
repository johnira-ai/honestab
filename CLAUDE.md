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
  test_analyze.py      # reference-value and input-validation unit tests
scripts/
  calibration_figure.py  # regenerates the README chart in docs/images/
.github/workflows/tests.yml  # CI: lint, format, mypy, build; tests on a
                             # Python/OS matrix and on lowest dependencies
.github/dependabot.yml       # monthly updates for actions and uv.lock
```

## Tooling

- Package and environment management: `uv` (Python 3.11+)
- Install: `uv sync` (dependency groups: `test`, `lint`, `docs`; `dev` includes all)
- Lint: `uv run ruff check` (config in `pyproject.toml`, line length 88)
- Format: `uv run ruff format`
- Type check: `uv run mypy` (strict, covers `src` and `tests`)
- Test: `uv run pytest` (warnings are errors)
- After changing dependencies, run `uv lock`; CI uses `uv sync --locked`.

## The rule

**Every new statistical method must ship with a calibration test that checks it
against simulated ground truth.** A method without a calibration test does not
get merged. At minimum, the test should simulate many experiments under a known
effect (including the null) and verify that the method's error rates and
interval coverage match their nominal levels within Monte Carlo tolerance.
