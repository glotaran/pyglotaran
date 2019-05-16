import numpy as np

from glotaran.model import model
from glotaran.models.spectral_temporal import (
    KineticModel,
    KineticMegacomplex,
    SpectralTemporalDatasetDescriptor,
)
from glotaran.models.spectral_temporal.kinetic_matrix import calculate_kinetic_matrix
from glotaran.models.spectral_temporal.kinetic_result import finalize_kinetic_data


def kinetic_image_matrix(dataset_descriptor: SpectralTemporalDatasetDescriptor = None,
                         axis: np.ndarray = None):
    return calculate_kinetic_matrix(dataset_descriptor, axis, 0)


@model(
    'kinetic_image',
    dataset_type=SpectralTemporalDatasetDescriptor,
    megacomplex_type=KineticMegacomplex,
    matrix=kinetic_image_matrix,
    matrix_dimension='time',
    global_dimension='pixel',
    grouped=False,
    finalize_data_function=finalize_kinetic_data,
)
class KineticImageModel(KineticModel):
    """."""
