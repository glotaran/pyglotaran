import typing
import numpy as np

from glotaran.model import model
from glotaran.parameter import ParameterGroup

from glotaran.builtin.models.kinetic_image.kinetic_image_megacomplex import KineticImageMegacomplex
from glotaran.builtin.models.kinetic_image.kinetic_image_model import KineticImageModel

from .kinetic_spectrum_dataset_descriptor import KineticSpectrumDatasetDescriptor
from .kinetic_spectrum_result import finalize_kinetic_spectrum_result
from .kinetic_spectrum_matrix import kinetic_spectrum_matrix
from .spectral_constraints import SpectralConstraint, apply_spectral_constraints
from .spectral_irf import IrfSpectralMultiGaussian
from .spectral_matrix import spectral_matrix
from .spectral_penalties import EqualAreaPenalty, has_spectral_penalties, apply_spectral_penalties
from .spectral_relations import SpectralRelation, apply_spectral_relations
from .spectral_shape import SpectralShape


def has_kinetic_model_constraints(model: typing.Type['KineticModel']) -> bool:
    return len(model.spectral_relations) + len(model.spectral_constraints) != 0


def apply_kinetic_model_constraints(
        model: typing.Type['KineticModel'],
        parameter: ParameterGroup,
        clp_labels: typing.List[str],
        matrix: np.ndarray,
        index: float):
    clp_labels, matrix = apply_spectral_relations(model, parameter, clp_labels, matrix, index)
    clp_labels, matrix = apply_spectral_constraints(model, clp_labels, matrix, index)
    return clp_labels, matrix


def index_dependend(model: typing.Type['KineticModel']):
    if any([
        isinstance(irf, IrfSpectralMultiGaussian) and irf.dispersion_center is not None
        for irf in model.irf.values()
    ]):
        return True
    if len(model.spectral_relations) != 0:
        return True
    if len(model.spectral_constraints) != 0:
        return True
    return False


def grouped(model: typing.Type['KineticModel']):
    return len(model.dataset) != 1


@model(
    'kinetic-spectrum',
    attributes={
        'equal_area_penalties': EqualAreaPenalty,
        'shape': SpectralShape,
        'spectral_constraints': SpectralConstraint,
        'spectral_relations': SpectralRelation,
    },
    dataset_type=KineticSpectrumDatasetDescriptor,
    megacomplex_type=KineticImageMegacomplex,
    matrix=kinetic_spectrum_matrix,
    matrix_dimension='time',
    global_matrix=spectral_matrix,
    global_dimension='spectral',
    has_matrix_constraints_function=has_kinetic_model_constraints,
    constrain_matrix_function=apply_kinetic_model_constraints,
    has_additional_penalty_function=has_spectral_penalties,
    additional_penalty_function=apply_spectral_penalties,
    grouped=grouped,
    index_dependend=index_dependend,
    finalize_data_function=finalize_kinetic_spectrum_result,
)
class KineticSpectrumModel(KineticImageModel):
    pass
