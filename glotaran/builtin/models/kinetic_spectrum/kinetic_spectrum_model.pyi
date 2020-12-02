from typing import Any
from typing import List
from typing import Mapping
from typing import Tuple
from typing import Union

import numpy as np

from glotaran.model import model
from glotaran.parameter import ParameterGroup

from ..kinetic_image.kinetic_image_megacomplex import KineticImageMegacomplex
from ..kinetic_image.kinetic_image_model import KineticImageModel
from .kinetic_spectrum_dataset_descriptor import KineticSpectrumDatasetDescriptor
from .kinetic_spectrum_matrix import kinetic_spectrum_matrix
from .kinetic_spectrum_result import finalize_kinetic_spectrum_result
from .spectral_constraints import SpectralConstraint
from .spectral_constraints import apply_spectral_constraints
from .spectral_irf import IrfSpectralMultiGaussian
from .spectral_matrix import spectral_matrix
from .spectral_penalties import EqualAreaPenalty
from .spectral_penalties import apply_spectral_penalties
from .spectral_penalties import has_spectral_penalties
from .spectral_relations import SpectralRelation
from .spectral_relations import apply_spectral_relations
from .spectral_relations import retrieve_related_clps
from .spectral_shape import SpectralShape

def has_kinetic_model_constraints(model: KineticSpectrumModel) -> bool:
    ...


def apply_kinetic_model_constraints(
    model: KineticSpectrumModel,
    parameter: ParameterGroup,
    clp_labels: List[str],
    matrix: np.ndarray,
    index: float,
) -> Any:
    ...


def retrieve_spectral_clps(
    model: KineticSpectrumModel,
    parameter: ParameterGroup,
    clp_labels: List[str],
    reduced_clp_labels: List[str],
    reduced_clps: Union[np.ndarray, List[np.ndarray]],
    global_axis: np.ndarray,
) -> Any:
    ...


def index_dependent(model: KineticSpectrumModel) -> Any:
    ...


def grouped(model: KineticSpectrumModel) -> bool:
    ...


class KineticSpectrumModel(KineticImageModel):
    dataset: Mapping[str, KineticSpectrumDatasetDescriptor]
    megacomplex: Mapping[str, KineticImageMegacomplex]

    @property
    def equal_area_penalties(self) -> List[EqualAreaPenalty]:
        ...

    @property
    def shape(self) -> Mapping[str, SpectralShape]:
        ...

    @property
    def spectral_constraints(self) -> List[SpectralConstraint]:
        ...

    @property
    def spectral_relations(self) -> List[SpectralRelation]:
        ...

    def has_matrix_constraints_function(self) -> bool:
        ...

    def constrain_matrix_function(
        self, parameter: ParameterGroup, clp_labels: List[str], matrix: np.ndarray, index: float
    ) -> Tuple[List[str], np.ndarray]:
        ...

    def retrieve_clp_function(
        self,
        parameter: ParameterGroup,
        clp_labels: List[str],
        reduced_clp_labels: List[str],
        reduced_clps: Union[np.ndarray, List[np.ndarray]],
        global_axis: np.ndarray,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        ...

    def has_additional_penalty_function(self) -> bool:
        ...

    def additional_penalty_function(
        self,
        parameter: ParameterGroup,
        clp_labels: Union[List[str], List[List[str]]],
        clps: np.ndarray,
        global_axis: np.ndarray,
    ) -> np.ndarray:
        ...

    @staticmethod
    def global_matrix(dataset, axis) -> Union[Tuple[None, None], Tuple[List[str], np.ndarray]]:  # type: ignore[override]
        ...

    @staticmethod
    def matrix(  # type: ignore[override]
        dataset_descriptor: KineticSpectrumDatasetDescriptor = None,
        axis=None,
        index=None,
        irf=None,
    ) -> Union[Tuple[None, None], Tuple[List[Any], np.ndarray]]:
        ...
