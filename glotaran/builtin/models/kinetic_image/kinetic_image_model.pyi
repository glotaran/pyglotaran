from __future__ import annotations

from typing import Any
from typing import Mapping

import numpy as np

from glotaran.model import Model
from glotaran.model import model  # noqa: F401

from .initial_concentration import InitialConcentration
from .irf import Irf
from .k_matrix import KMatrix
from .kinetic_image_dataset_descriptor import KineticImageDatasetDescriptor
from .kinetic_image_matrix import kinetic_image_matrix  # noqa: F401
from .kinetic_image_megacomplex import KineticImageMegacomplex
from .kinetic_image_result import finalize_kinetic_image_result  # noqa: F401

class KineticImageModel(Model):
    dataset: Mapping[str, KineticImageDatasetDescriptor]
    megacomplex: Mapping[str, KineticImageMegacomplex]
    @staticmethod
    def matrix(  # type: ignore[override]
        dataset_descriptor: KineticImageDatasetDescriptor = ..., axis=..., index=..., irf=...
    ) -> tuple[None, None] | tuple[list[Any], np.ndarray]: ...
    @property
    def initial_concentration(self) -> Mapping[str, InitialConcentration]: ...
    @property
    def k_matrix(self) -> Mapping[str, KMatrix]: ...
    @property
    def irf(self) -> Mapping[str, Irf]: ...
