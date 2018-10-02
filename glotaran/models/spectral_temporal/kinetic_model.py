"""Glotaran Kinetic Model"""

from glotaran.model import BaseModel
from glotaran.model import model
# from glotaran.model.model import model

from .initial_concentration import InitialConcentration
from .irf import Irf
from .k_matrix import KMatrix
from .kinetic_megacomplex import KineticMegacomplex
from .spectral_shape import SpectralShape
from .spectral_temporal_dataset_descriptor import SpectralTemporalDatasetDescriptor
from .kinetic_matrix import calculate_kinetic_matrix
from .spectral_matrix import calculate_spectral_matrix


@model(
    'kinetic',
    attributes={
        'initial_concentration': InitialConcentration,
        'k_matrix': KMatrix,
        'irf': Irf,
        'shape': SpectralShape,
    },
    dataset_type=SpectralTemporalDatasetDescriptor,
    megacomplex_type=KineticMegacomplex,
    calculated_matrix=calculate_kinetic_matrix,
    calculated_axis='time',
    estimated_matrix=calculate_spectral_matrix,
    estimated_axis='spectral',
)
class KineticModel(BaseModel):
    """A kinetic model is an implementation for model.Model. It is used describe
    time dependend datasets.

    """
