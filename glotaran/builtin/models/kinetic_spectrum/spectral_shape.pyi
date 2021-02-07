from __future__ import annotations

import numpy as np

from glotaran.model import model_attribute  # noqa: F401
from glotaran.model import model_attribute_typed  # noqa: F401
from glotaran.parameter import Parameter

class SpectralShapeGaussian:
    @property
    def amplitude(self) -> Parameter: ...
    @property
    def location(self) -> Parameter: ...
    @property
    def width(self) -> Parameter: ...
    def calculate(self, axis: np.ndarray) -> np.ndarray: ...

class SpectralShapeOne:
    def calculate(self, axis: np.ndarray) -> np.ndarray: ...

class SpectralShapeZero:
    def calculate(self, axis: np.ndarray) -> np.ndarray: ...

class SpectralShape:
    @classmethod
    def add_type(cls, type_name: str, attribute_type: type) -> None: ...
