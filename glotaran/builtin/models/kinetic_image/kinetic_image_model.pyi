from __future__ import annotations

from typing import Any
from typing import Mapping

import numpy as np

from glotaran.builtin.models.kinetic_image.initial_concentration import InitialConcentration
from glotaran.builtin.models.kinetic_image.irf import Irf
from glotaran.builtin.models.kinetic_image.k_matrix import KMatrix
from glotaran.builtin.models.kinetic_image.kinetic_image_dataset_descriptor import (
    KineticImageDatasetDescriptor,
)
from glotaran.builtin.models.kinetic_image.kinetic_image_matrix import kinetic_image_matrix
from glotaran.builtin.models.kinetic_image.kinetic_image_megacomplex import KineticImageMegacomplex
from glotaran.builtin.models.kinetic_image.kinetic_image_result import (
    finalize_kinetic_image_result,
)
from glotaran.model import Model
from glotaran.model import model

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
