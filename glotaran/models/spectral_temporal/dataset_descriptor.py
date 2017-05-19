from glotaran.model import DatasetDescriptor


class SpectralTemporalDatasetDescriptor(DatasetDescriptor):
    """SpectralTemporalDatasetDescriptor is an implementation of
    model.DatasetDescriptor for spectral or temporal models.

    A SpectralTemporalDatasetDescriptor additionally contains an
    instrument response functions(Irf) and one or more spectral shapes.
    """
    def __init__(self, label, initial_concentration, megacomplexes,
                 megacomplex_scalings, dataset_scaling,
                 compartment_scalings, compartment_constraints, irf, shapes):
        self.irf = irf
        self.shapes = shapes
        super(SpectralTemporalDatasetDescriptor, self).\
            __init__(label,
                     initial_concentration,
                     megacomplexes,
                     megacomplex_scalings,
                     dataset_scaling,
                     compartment_scalings,
                     compartment_constraints)

    @property
    def irf(self):
        """label of the Irf to be used to fit the dataset."""
        return self._irf

    @irf.setter
    def irf(self, irf):
        """

        Parameters
        ----------
        irf : label of the Irf to be used to fit the dataset.

        """
        self._irf = irf

    @property
    def shapes(self):
        """list of shape labels"""
        return self._shapes

    @shapes.setter
    def shapes(self, value):
        """

        Parameters
        ----------
        value : list of shape labels


        Returns
        -------

        """
        self._shapes = value

    def __str__(self):
        return "{}\n\tIrf: {}".format(super(SpectralTemporalDatasetDescriptor, self)
                                      .__str__(),
                                      self.irf)
