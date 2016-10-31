from glotaran_core.model import DatasetDescriptor
from .irf import Irf


class KineticDatasetDescriptor(DatasetDescriptor):
    """
    KineticDataset is a Dataset with an Irf
    """
    def __init__(self, label, initial_concentration, megacomplexes,
                 megacomplex_scalings, dataset, dataset_scaling, irf):
        self.irf = irf
        super(KineticDatasetDescriptor, self).__init__(label,
                                                       initial_concentration,
                                                       megacomplexes,
                                                       megacomplex_scalings,
                                                       dataset,
                                                       dataset_scaling)

    @property
    def irf(self):
        return self._irf

    @irf.setter
    def set_irf(self, irf):
        if not issubclass(type(irf), Irf):
            raise TypeError("Irf musst be subclass of 'Irf'.")
        self._irf = irf

    def __str__(self):
        return "{}\n\t\tIRF: {}".format(super(KineticDatasetDescriptor, self)
                                        .__str__(),
                                        self.irf())
