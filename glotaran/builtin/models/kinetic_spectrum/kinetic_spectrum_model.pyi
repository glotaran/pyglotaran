from __future__ import annotations

from typing import Any
from typing import Mapping

import numpy as np

from glotaran.builtin.models.kinetic_image.kinetic_image_megacomplex import KineticImageMegacomplex
from glotaran.builtin.models.kinetic_image.kinetic_image_model import KineticImageModel
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_dataset_descriptor import (
    KineticSpectrumDatasetDescriptor,
)
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_matrix import (
    kinetic_spectrum_matrix,
)
from glotaran.builtin.models.kinetic_spectrum.kinetic_spectrum_result import (
    finalize_kinetic_spectrum_result,
)
from glotaran.builtin.models.kinetic_spectrum.spectral_constraints import SpectralConstraint
from glotaran.builtin.models.kinetic_spectrum.spectral_constraints import (
    apply_spectral_constraints,
)
from glotaran.builtin.models.kinetic_spectrum.spectral_irf import IrfSpectralMultiGaussian
from glotaran.builtin.models.kinetic_spectrum.spectral_matrix import spectral_matrix
from glotaran.builtin.models.kinetic_spectrum.spectral_penalties import EqualAreaPenalty
from glotaran.builtin.models.kinetic_spectrum.spectral_penalties import apply_spectral_penalties
from glotaran.builtin.models.kinetic_spectrum.spectral_penalties import has_spectral_penalties
from glotaran.builtin.models.kinetic_spectrum.spectral_relations import SpectralRelation
from glotaran.builtin.models.kinetic_spectrum.spectral_relations import apply_spectral_relations
from glotaran.builtin.models.kinetic_spectrum.spectral_relations import retrieve_related_clps
from glotaran.builtin.models.kinetic_spectrum.spectral_shape import SpectralShape
from glotaran.model import model
from glotaran.parameter import ParameterGroup

def has_kinetic_model_constraints(model: KineticSpectrumModel) -> bool: ...  # noqa: F811
def apply_kinetic_model_constraints(
    model: KineticSpectrumModel,  # noqa: F811
    parameters: ParameterGroup,
    clp_labels: list[str],
    matrix: np.ndarray,
    index: float,
) -> Any: ...
def retrieve_spectral_clps(
    model: KineticSpectrumModel,  # noqa: F811
    parameters: ParameterGroup,
    clp_labels: list[str],
    reduced_clp_labels: list[str],
    reduced_clps: np.ndarray | list[np.ndarray],
    global_axis: np.ndarray,
) -> Any: ...
def index_dependent(model: KineticSpectrumModel) -> Any: ...  # noqa: F811
def grouped(model: KineticSpectrumModel) -> bool: ...  # noqa: F811

class KineticSpectrumModel(KineticImageModel):
    dataset: Mapping[str, KineticSpectrumDatasetDescriptor]
    megacomplex: Mapping[str, KineticImageMegacomplex]
    @property
    def equal_area_penalties(self) -> list[EqualAreaPenalty]: ...
    @property
    def shape(self) -> Mapping[str, SpectralShape]: ...
    @property
    def spectral_constraints(self) -> list[SpectralConstraint]: ...
    @property
    def spectral_relations(self) -> list[SpectralRelation]: ...
    def has_matrix_constraints_function(self) -> bool: ...
    def constrain_matrix_function(
        self, parameters: ParameterGroup, clp_labels: list[str], matrix: np.ndarray, index: float
    ) -> tuple[list[str], np.ndarray]: ...
    def retrieve_clp_function(
        self,
        parameters: ParameterGroup,
        clp_labels: list[str],
        reduced_clp_labels: list[str],
        reduced_clps: np.ndarray | list[np.ndarray],
        global_axis: np.ndarray,
    ) -> np.ndarray | list[np.ndarray]: ...
    def has_additional_penalty_function(self) -> bool: ...
    def additional_penalty_function(
        self,
        parameters: ParameterGroup,
        clp_labels: list[str] | list[list[str]],
        clps: np.ndarray,
        global_axis: np.ndarray,
    ) -> np.ndarray: ...
    @staticmethod
    def global_matrix(dataset, axis) -> tuple[None, None] | tuple[list[str], np.ndarray]: ...
    @staticmethod
    def matrix(  # type: ignore[override]
        dataset_descriptor: KineticSpectrumDatasetDescriptor = ...,
        axis=...,
        index=...,
        irf=...,
    ) -> tuple[None, None] | tuple[list[Any], np.ndarray]: ...
