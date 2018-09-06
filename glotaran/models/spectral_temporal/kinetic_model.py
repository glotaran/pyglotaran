"""Glotaran Kinetic Model"""

from typing import Type, Dict
from glotaran.model import Model
from glotaran.model import glotaran_model

from .irf import Irf
from .k_matrix import KMatrix
from .kinetic_megacomplex import KineticMegacomplex
from .spectral_shape import SpectralShape
from .spectral_temporal_dataset_descriptor import SpectralTemporalDatasetDescriptor
from .kinetic_matrix import calculate_kinetic_matrix
from .spectral_matrix import calculate_spectral_matrix


@glotaran_model('kinetic',
                attributes={
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
class KineticModel(Model):
    """A kinetic model is an implementation for model.Model. It is used describe
    time dependend datasets.

    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=attribute-defined-outside-init
    # Models are complex.


    def get_megacomplex_k_matrix(self, cmplx: str) -> KMatrix:
        cmplx = self.megacomplexes[cmplx]
        kmat = KMatrix.empty(cmplx.label, self.compartments)
        for mat in cmplx.k_matrices:
            kmat = kmat.combine(self.k_matrices[mat])
        return kmat
