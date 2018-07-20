from math import isnan

from lmfit import Parameter as LmParameter


class Parameter(LmParameter):
    """Wrapper for lmfit parameter."""
    def __init__(self):
        self.index = -1
        self.fit = True
        self.label = None
        super(Parameter, self).__init__()

    @property
    def index(self):
        """Index in the parameter tree"""
        return self._index

    @index.setter
    def index(self, i):
        """

        Parameters
        ----------
        i : index


        Returns
        -------


        """
        self._index = i

    @property
    def label(self):
        """Label of the parameter"""
        return self._label

    @label.setter
    def label(self, label):
        """

        Parameters
        ----------
        label : label of the parameter


        Returns
        -------


        """
        self._label = label

    @property
    def fit(self):
        """True or false"""
        return self._fit

    @fit.setter
    def fit(self, value):
        """

        Parameters
        ----------
        value : true or false


        Returns
        -------


        """
        if not isinstance(value, bool):
            raise TypeError("Fit must be True or False")
        self._fit = value

    @LmParameter.value.setter
    def value(self, val):
        """

        Parameters
        ----------
        val : value of the parameter


        Returns
        -------


        """

        if not isinstance(val, (int, float)):
                try:
                    val = float(val)
                except:
                    raise Exception("Parameter Error: value must be numeric:"
                                    "{} Type: {}".format(val, type(val)))

        if isinstance(val, int):
            val = float(val)

        if isnan(val):
            self.vary = False

        LmParameter.value.fset(self, val)

    def __str__(self):
        """ """
        return f"**{self.label}**:\t _Value_: {self.value}\t_Min_:" + \
               f" {self.min}\t_Max_: {self.max}\t_Vary_: {self.vary} _Fit_: {self.fit}"
