"""honestab: A/B tests with honest error bars."""

from importlib.metadata import version

from honestab.analyze import AnalysisResult, analyze
from honestab.planning import minimum_detectable_effect, power, sample_size
from honestab.simulate import simulate
from honestab.srm import SRMResult, srm_check

__all__ = [
    "AnalysisResult",
    "SRMResult",
    "analyze",
    "minimum_detectable_effect",
    "power",
    "sample_size",
    "simulate",
    "srm_check",
]
__version__ = version("honestab")
