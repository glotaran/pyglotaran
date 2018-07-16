"""Spectral Temporal Dataset Descriptor"""

from typing import Dict, List
from glotaran.model import DatasetDescriptor

from .compartment_constraints import CompartmentConstraint


class SpectralTemporalDatasetDescriptor(DatasetDescriptor):
    """SpectralTemporalDatasetDescriptor is an implementation of
    model.DatasetDescriptor for spectral or temporal models.

    A SpectralTemporalDatasetDescriptor additionally contains an
    instrument response functions(IRF) and one or more spectral shapes.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=attribute-defined-outside-init
    # Datasets are complex.

    def __init__(self,
                 label: str,
                 initial_concentration: str,
                 megacomplexes: List[str],
                 megacomplex_scaling: Dict[str, List[str]],
                 scaling: str,
                 compartment_scaling: Dict[str, List[str]],
                 compartment_constraints: List[CompartmentConstraint],
                 irf: str,
                 shapes: Dict[str, str]):
        """

        Parameters
        ----------
        label : str
            The label of the dataset.

        initial_concentration : str
            The label of the initial concentration

        megacomplexes : List[str]
            A list of megacomplex labels

        megacomplex_scaling : Dict[str: List[str]]
            The megacomplex scaling parameters

        scaling : str
            The scaling parameter for the dataset

        compartment_scaling: Dict[str: List[str]]
            The compartment scaling parameters

        compartment_constraints: List[CompartmentConstraint] :
            A list of compartment constraints

        irf : str
            The label of the dataset's IRF

        shapes : dict(str, str)
            A dictory of comparment and shape label pairs

        """
        self.irf = irf
        self.shapes = shapes
        super(SpectralTemporalDatasetDescriptor, self).\
            __init__(label,
                     initial_concentration,
                     megacomplexes,
                     megacomplex_scaling,
                     scaling,
                     compartment_scaling,
                     compartment_constraints)

    @property
    def irf(self) -> str:
        """The label of the dataset's IRF"""
        return self._irf

    @irf.setter
    def irf(self, irf):
        self._irf = irf

    @property
    def shapes(self) -> Dict[str, str]:
        """A dictory of comparment and shape label pairs"""
        return self._shapes

    @shapes.setter
    def shapes(self, value):
        self._shapes = value

    def __str__(self):
        return "{}\n\tIrf: {}".format(super(SpectralTemporalDatasetDescriptor, self)
                                      .__str__(),
                                      self.irf)
