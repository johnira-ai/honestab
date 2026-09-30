"""honestab: A/B tests with honest error bars."""

from importlib.metadata import version

from honestab.analyze import AnalysisResult, analyze
from honestab.simulate import simulate

__all__ = ["AnalysisResult", "analyze", "simulate"]
__version__ = version("honestab")
