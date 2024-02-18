from .biaffine import BiaffineModel as DependencyParsingModel
from .rel_predictor import RelModel

__all__ = [
    "DependencyParsingModel",
    "RelModel"
]
