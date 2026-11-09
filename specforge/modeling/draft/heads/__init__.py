"""Algorithm heads used by composed draft models."""

from .adjacent_injection import AdjacentInjection
from .candidate_selector import CandidateSelector
from .confidence import AcceptRatePredictor
from .domino_gru import DominoGRUHead
from .markov import GatedMarkovHead, RNNMarkovHead, VanillaMarkovHead

__all__ = [
    "DominoGRUHead",
    "AdjacentInjection",
    "CandidateSelector",
    "AcceptRatePredictor",
    "GatedMarkovHead",
    "RNNMarkovHead",
    "VanillaMarkovHead",
]
