class Irf(object):
    """Represents an IRF."""

    def __init__(self, label):
        self.label = label

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

    def type_string(self):
        """Identifies an implementation """
        raise NotImplementedError

    def __str__(self):
        return f"### _{self.label}_\n* _Type_: {self.type_string()}\n"
