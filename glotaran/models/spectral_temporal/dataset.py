from glotaran.model import Dataset


class SpectralTemporalDataset(Dataset):
    """Implementation of model.Dataset for spectral and temporal models.
    Provides a time and a spectral axis. """

    def __init__(self, label):
        super(SpectralTemporalDataset, self).__init__(label)

    @property
    def time_axis(self):
        """the time axis"""
        return self.get_axis("time")

    @time_axis.setter
    def time_axis(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        self.set_axis("time", value)

    @property
    def spectral_axis(self):
        """spec spectral axis """
        return self.get_axis("spectral")

    @spectral_axis.setter
    def spectral_axis(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        self.set_axis("spectral", value)

    def get_estimated_axis(self):
        """ """
        return self.spectral_axis

    def get_calculated_axis(self):
        """ """
        return self.time_axis
