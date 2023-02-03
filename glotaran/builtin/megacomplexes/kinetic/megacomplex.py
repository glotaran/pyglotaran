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

    def calculate_rates_and_a_matrices(
        self, model: ActivationDataModel
    ) -> tuple[list[np.typing.ArrayLike], list[np.typing.ArrayLike]]:
        kinetic = Kinetic.combine(self.kinetic)
        rates = []
        a_matrices = []
        for activation in model.activation:
            initial_concentrations = np.array(
                [activation.compartments.get(label, 0) for label in kinetic.compartments]
            )
            rates.append(kinetic.rates(initial_concentrations))
            a_matrices.append(kinetic.a_matrix(initial_concentrations))
        return rates, a_matrices

    def calculate_matrix(
        self,
        model: ActivationDataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
    ) -> tuple[list[str], np.typing.ArrayLike]:
        rates, a_matrices = self.calculate_rates_and_a_matrices(model)
