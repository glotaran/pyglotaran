from glotaran.model import DatasetDescriptor


class KineticDatasetDescriptor(DatasetDescriptor):
    """
    KineticDataset is a Dataset with an Irf
    """
    def __init__(self, label, initial_concentration, megacomplexes,
                 megacomplex_scalings, dataset_scaling, irf):
        self.irf = irf
        super(KineticDatasetDescriptor, self).__init__(label,
                                                       initial_concentration,
                                                       megacomplexes,
                                                       megacomplex_scalings,
                                                       dataset_scaling)

    @property
    def irf(self):
        '''Returns the label for the Irf to be used to fit the dataset.'''
        return self._irf

    @irf.setter
    def irf(self, irf):
        '''Sets the label for the Irf to be used to fit the dataset.'''
        self._irf = irf

    def __str__(self):
        return "{}\n\tIrf: {}".format(super(KineticDatasetDescriptor, self)
                                      .__str__(),
                                      self.irf)
