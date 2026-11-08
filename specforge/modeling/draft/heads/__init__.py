"""Algorithm heads shared by legacy and composed draft models."""

from .adjacent_injection import AdjacentInjection
from .candidate_selector import CandidateSelector
from .confidence import AcceptRatePredictor
from .markov import GatedMarkovHead, RNNMarkovHead, VanillaMarkovHead

__all__ = [
    "AdjacentInjection",
    "CandidateSelector",
    "AcceptRatePredictor",
    "GatedMarkovHead",
    "RNNMarkovHead",
    "VanillaMarkovHead",
]
