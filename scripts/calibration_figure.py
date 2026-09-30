"""Generate the calibration figure shown in the README.

Simulates many A/B experiments with a known true effect, analyzes each one with
honestab, and plots two checks side by side:

* confidence interval coverage for a binary metric with a rare event, across
  sample sizes, for honestab's Newcombe interval and the textbook Wald interval;
* the distribution of Welch's t-test p-values in A/A tests, which should be
  flat. (Binary p-values are discrete, so their histogram is lumpy near 1 even
  when the test is calibrated; the continuous metric shows the shape cleanly.)

Usage::

    uv run python scripts/calibration_figure.py

Writes ``docs/images/calibration-light.png`` and ``calibration-dark.png``. Seeds
are fixed, so the output is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy import stats

from honestab import AnalysisResult, analyze, simulate
from honestab.simulate import Metric

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"

ALPHA = 0.05
Z_TOL = 4.0  # same Monte Carlo tolerance as tests/test_calibration.py

# Left panel: coverage of the binary interval in a rare-event setting.
COVERAGE_SIZES = [25, 50, 100, 200, 500, 1_000, 2_000]
COVERAGE_BASELINE = 0.02
COVERAGE_EFFECT = 0.01
COVERAGE_SIMS = 4_000

# Right panel: p-values under the null, continuous metric.
NULL_N_PER_ARM = 1_000
NULL_MEAN = 50.0
NULL_SD = 20.0
NULL_SIMS = 10_000
NULL_BINS = 20


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str  # the page the image sits on; used for marker rings
    ink: str
    secondary: str
    muted: str
    grid: str
    baseline: str
    newcombe: str
    wald: str
    band: str


LIGHT = Theme(
    name="light",
    surface="#ffffff",
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    newcombe="#2a78d6",
    wald="#eb6834",
    band="#0b0b0b",
)
DARK = Theme(
    name="dark",
    surface="#0d1117",
    ink="#ffffff",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    newcombe="#3987e5",
    wald="#d95926",
    band="#ffffff",
)


def run(
    seed: int,
    n_sims: int,
    n_per_arm: int,
    metric: Metric,
    baseline: float,
    effect: float = 0.0,
    sd: float = 1.0,
) -> list[AnalysisResult]:
    rngs = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_sims)
    ]
    return [
        analyze(
            simulate(
                n_per_arm,
                seed=rng,
                metric=metric,
                baseline=baseline,
                effect=effect,
                sd=sd,
            ),
            alpha=ALPHA,
        )
        for rng in rngs
    ]


def wald_covers(result: AnalysisResult, n: int, effect: float) -> bool:
    """Whether the textbook Wald interval covers ``effect``, for comparison."""
    p_c, p_t = result.control_mean, result.treatment_mean
    se = np.sqrt(p_c * (1 - p_c) / n + p_t * (1 - p_t) / n)
    z = stats.norm.ppf(1 - ALPHA / 2)
    return bool(abs(result.effect - effect) <= z * se)


def coverage_by_size() -> tuple[np.ndarray, np.ndarray]:
    newcombe, wald = [], []
    for i, n in enumerate(COVERAGE_SIZES):
        results = run(
            seed=100 + i,
            n_sims=COVERAGE_SIMS,
            n_per_arm=n,
            metric="binary",
            baseline=COVERAGE_BASELINE,
            effect=COVERAGE_EFFECT,
        )
        newcombe.append(
            np.mean([r.ci_low <= COVERAGE_EFFECT <= r.ci_high for r in results])
        )
        wald.append(np.mean([wald_covers(r, n, COVERAGE_EFFECT) for r in results]))
        print(f"n={n:>5}: newcombe {newcombe[-1]:.3f}  wald {wald[-1]:.3f}")
    return np.array(newcombe), np.array(wald)


def null_p_values() -> np.ndarray:
    results = run(
        seed=200,
        n_sims=NULL_SIMS,
        n_per_arm=NULL_N_PER_ARM,
        metric="continuous",
        baseline=NULL_MEAN,
        sd=NULL_SD,
    )
    return np.array([r.p_value for r in results])


def style_axes(ax: Axes, theme: Theme) -> None:
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.baseline)
    ax.tick_params(colors=theme.muted, length=0, labelsize=10, pad=6)
    ax.grid(axis="y", color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)


def title(ax: Axes, theme: Theme, heading: str, subtitle: str) -> None:
    ax.text(
        0,
        1.13,
        heading,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="semibold",
        color=theme.ink,
    )
    ax.text(
        0, 1.05, subtitle, transform=ax.transAxes, fontsize=10, color=theme.secondary
    )


def plot_coverage(
    ax: Axes, theme: Theme, newcombe: np.ndarray, wald: np.ndarray
) -> None:
    style_axes(ax, theme)
    nominal = 1 - ALPHA
    tol = Z_TOL * np.sqrt(nominal * (1 - nominal) / COVERAGE_SIMS)
    ax.axhspan(nominal - tol, nominal + tol, color=theme.band, alpha=0.06, lw=0)
    ax.axhline(nominal, color=theme.muted, linewidth=1)
    ax.text(
        COVERAGE_SIZES[-1],
        nominal + tol + 0.004,
        "nominal 95%",
        ha="right",
        va="bottom",
        fontsize=9,
        color=theme.muted,
    )

    for values, color, label in (
        (wald, theme.wald, "Wald (textbook)"),
        (newcombe, theme.newcombe, "Newcombe (honestab)"),
    ):
        ax.plot(
            COVERAGE_SIZES,
            values,
            color=color,
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            marker="o",
            markersize=8,
            markeredgecolor=theme.surface,
            markeredgewidth=2,
            label=label,
            zorder=3,
        )

    # Direct labels where the lines are furthest apart and free of other text.
    for values, text, i, offset, va in (
        (newcombe, "Newcombe", 2, 0.008, "bottom"),
        (wald, "Wald", 1, -0.012, "top"),
    ):
        ax.text(
            COVERAGE_SIZES[i] * 1.08,
            values[i] + offset,
            text,
            ha="left",
            va=va,
            fontsize=10,
            color=theme.secondary,
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(COVERAGE_SIZES))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlim(COVERAGE_SIZES[0] / 1.25, COVERAGE_SIZES[-1] * 1.25)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlabel("users per arm", color=theme.secondary, fontsize=10, labelpad=8)

    legend = ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=10,
        labelcolor=theme.secondary,
        handlelength=1.6,
    )
    legend.set_zorder(4)
    title(
        ax,
        theme,
        "Confidence interval coverage",
        f"Binary metric, {COVERAGE_BASELINE:.0%} baseline, "
        f"+{COVERAGE_EFFECT * 100:.0f} pt true effect",
    )


def plot_null_p_values(ax: Axes, theme: Theme, p_values: np.ndarray) -> None:
    style_axes(ax, theme)
    edges = np.linspace(0, 1, NULL_BINS + 1)
    counts, _ = np.histogram(p_values, bins=edges)
    expected = NULL_SIMS / NULL_BINS
    tol = Z_TOL * np.sqrt(NULL_SIMS * (1 / NULL_BINS) * (1 - 1 / NULL_BINS))

    ax.axhspan(expected - tol, expected + tol, color=theme.band, alpha=0.06, lw=0)
    width = edges[1] - edges[0]
    ax.bar(
        edges[:-1],
        counts,
        width=width,
        align="edge",
        color=theme.newcombe,
        edgecolor=theme.surface,
        linewidth=2,
        zorder=2,
    )
    ax.axhline(expected, color=theme.muted, linewidth=1, zorder=3)
    label_y = max(counts.max(), expected + tol) + 25
    ax.text(
        0.5,
        label_y,
        "expected if calibrated (flat)",
        ha="center",
        va="bottom",
        fontsize=9,
        color=theme.muted,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, label_y + 50)
    ax.xaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.set_xlabel("p-value", color=theme.secondary, fontsize=10, labelpad=8)
    title(
        ax,
        theme,
        "p-values in A/A tests",
        f"Welch's t-test, {NULL_SIMS:,} experiments with no true effect",
    )


def render(
    theme: Theme, newcombe: np.ndarray, wald: np.ndarray, p_values: np.ndarray
) -> Path:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.22}
    )
    fig.patch.set_alpha(0)
    plot_coverage(left, theme, newcombe, wald)
    plot_null_p_values(right, theme, p_values)
    fig.text(
        0.125,
        -0.04,
        f"Shaded bands: ±{Z_TOL:g} Monte Carlo standard errors. "
        f"Coverage uses {COVERAGE_SIMS:,} simulated experiments per sample size.",
        fontsize=9,
        color=theme.muted,
    )
    path = OUT_DIR / f"calibration-{theme.name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15, transparent=True)
    plt.close(fig)
    return path


def main() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ["Segoe UI", "Helvetica Neue", "Arial"]
    mpl.rcParams["font.family"] = [f for f in preferred if f in installed] + [
        "DejaVu Sans"
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    newcombe, wald = coverage_by_size()
    p_values = null_p_values()
    for theme in (LIGHT, DARK):
        print(f"wrote {render(theme, newcombe, wald, p_values)}")


if __name__ == "__main__":
    main()
