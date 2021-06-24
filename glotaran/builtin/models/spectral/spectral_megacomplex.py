from __future__ import annotations

from typing import Dict

import numpy as np

from glotaran.model import DatasetDescriptor
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import megacomplex


@megacomplex(
    "spectral",
    properties={
        "shape": Dict[str, str],
    },
)
class SpectralMegacomplex(Megacomplex):
    def calculate_matrix(
        self,
        model,
        dataset_model: DatasetDescriptor,
        indices: dict[str, int],
        axis: dict[str, np.ndarray],
        **kwargs,
    ):

        compartments = []
        for compartment in self.shape:
            if compartment in compartments:
                raise ModelError(f"More then one shape defined for compartment '{compartment}'")
            compartments.append(compartment)

        dim1 = axis[dataset_model.get_model_dimension()].size
        dim2 = len(self.shape)
        matrix = np.zeros((dim1, dim2))

        for i, shape in enumerate(self.shape.values()):
            matrix[:, i] += shape.calculate(axis[dataset_model.get_model_dimension()])
        return compartments, matrix
