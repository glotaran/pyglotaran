from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from glotaran.model import Model
from glotaran.model import model

from .initial_concentration import InitialConcentration
from .irf import Irf
from .k_matrix import KMatrix
from .kinetic_image_dataset_descriptor import KineticImageDatasetDescriptor
from .kinetic_image_matrix import kinetic_image_matrix
from .kinetic_image_megacomplex import KineticImageMegacomplex
from .kinetic_image_result import finalize_kinetic_image_result

class KineticImageModel(Model):
    dataset: Dict[str, KineticImageDatasetDescriptor]  # type: ignore[assignment]
    megacomplex: Dict[str, KineticImageMegacomplex]

    @staticmethod
    def matrix(  # type: ignore[override]
        dataset_descriptor: KineticImageDatasetDescriptor = None, axis=None, index=None, irf=None
    ) -> Union[Tuple[None, None], Tuple[List[Any], np.ndarray]]:
        ...

    @property
    def initial_concentration(self) -> Dict[str, InitialConcentration]:
        ...

    @property
    def k_matrix(self) -> Dict[str, KMatrix]:
        ...

    @property
    def irf(self) -> Dict[str, Irf]:
        ...
