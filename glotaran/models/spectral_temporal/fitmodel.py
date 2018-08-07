"""Glotaran Kinetic Fitmodel"""
from glotaran.fitmodel import FitModel

from .result import KineticResult


class KineticFitModel(FitModel):
    """Thin wrapper to return KineticResult instead of fitmodel.Result"""
    def result_class(self) -> KineticResult:
        """Returns KineticResult"""
        return KineticResult
