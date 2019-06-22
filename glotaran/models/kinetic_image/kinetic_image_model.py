from glotaran.model import model, Model

from .initial_concentration import InitialConcentration
from .irf import Irf
from .k_matrix import KMatrix
from .kinetic_image_dataset_descriptor import KineticImageDatasetDescriptor
from .kinetic_image_matrix import kinetic_image_matrix
from .kinetic_image_megacomplex import KineticImageMegacomplex
from .kinetic_image_result import finalize_kinetic_image_result


@model(
    'kinetic-image',
    attributes={
        'initial_concentration': InitialConcentration,
        'k_matrix': KMatrix,
        'irf': Irf,
    },
    dataset_type=KineticImageDatasetDescriptor,
    megacomplex_type=KineticImageMegacomplex,
    matrix=kinetic_image_matrix,
    matrix_dimension='time',
    global_dimension='pixel',
    grouped=False,
    finalize_data_function=finalize_kinetic_image_result,
)
class KineticImageModel(Model):
    pass
