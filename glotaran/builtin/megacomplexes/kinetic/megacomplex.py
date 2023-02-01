from typing import Literal

import numpy as np

from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.kinetic import Kinetic
from glotaran.model import LibraryItemType
from glotaran.model import Megacomplex


class KineticMegacomplex(Megacomplex):
    type: Literal["kinetic"]
    register_as = "kinetic"
    data_model = ActivationDataModel
    dimension: str = "time"
    kinetic: list[LibraryItemType[Kinetic]]

    def calculate_matrix(
        self,
        data_model: ActivationDataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
    ) -> tuple[list[str], np.typing.ArrayLike]:
        pass
