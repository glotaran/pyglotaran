class Irf(object):
    """Represents an IRF."""

    def __init__(self, label, backsweep=False, backsweep_period=0):
        self.label = label
        self.backsweep = backsweep
        self.backsweep_period = backsweep_period

    @property
    def label(self):
        """Label of the IRF"""
        return self._label

    @label.setter
    def label(self, value):
        """

        Parameters
        ----------
        value : label of the IRF


        Returns
        -------


        """
        if not isinstance(value, str):
            raise TypeError("Labels must be strings.")
        self._label = value

    @property
    def backsweep(self):
        """True or false """
        return self._backsweep

    @backsweep.setter
    def backsweep(self, value):
        """

        Parameters
        ----------
        value : True or False


        Returns
        -------


        """
        if not isinstance(value, bool):
            raise TypeError("Backsweep must be True or False")
        self._backsweep = value

    @property
    def backsweep_period(self):
        """Parameter Index"""
        return self._backsweep_period

    @backsweep_period.setter
    def backsweep_period(self, value):
        """

        Parameters
        ----------
        value : Parameter Index


        Returns
        -------


        """
        self._backsweep_period = value

    def type_string(self):
        """Identifies an implementation """
        raise NotImplementedError

    def __str__(self):
        return "Label: {} Type: {}".format(self.label, self.type_string())
