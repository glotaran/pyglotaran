from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from glotaran.builtin.models.kinetic_image.irf import Irf
from glotaran.builtin.models.kinetic_image.irf import IrfMultiGaussian
from glotaran.model import model_attribute
from glotaran.parameter import Parameter

class IrfSpectralMultiGaussian(IrfMultiGaussian):
    @property
    def dispersion_center(self) -> Parameter:
        ...

    @property
    def center_dispersion(self) -> List[Parameter]:
        ...

    @property
    def width_dispersion(self) -> List[Parameter]:
        ...

    @property
    def model_dispersion_with_wavenumber(self) -> bool:
        ...

    def parameter(self, index: Any):
        ...

    def calculate_dispersion(self, axis: Any):
        ...


class IrfSpectralGaussian(IrfSpectralMultiGaussian):
    @property
    def center(self) -> Parameter:
        ...

    @property
    def width(self) -> Parameter:
        ...


class IrfGaussianCoherentArtifact(IrfSpectralGaussian):
    @property
    def coherent_artifact_order(self) -> int:
        ...

    @property
    def coherent_artifact_width(self) -> Parameter:
        ...

    def clp_labels(self):
        ...

    def calculate_coherent_artifact(self, axis: Any):
        ...
