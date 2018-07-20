from math import isnan

from lmfit import Parameter as LmParameter


class Parameter(LmParameter):
    """Wrapper for lmfit parameter."""
    def __init__(self):
        self.index = -1
        self.fit = True
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
        return self.name

    @label.setter
    def label(self, label):
        """

        Parameters
        ----------
        label : label of the parameter


        Returns
        -------


        """
        self.name = label

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
                except ValueError:
                    raise Exception("Parameter Error: value must be numeric:"
                                    "{} Type: {}".format(val, type(val)))

        if isinstance(val, int):
            val = float(val)

        if isnan(val):
            self.vary = False

        LmParameter.value.fset(self, val)

    def _str__(self):
        """ """
        return 'Label: {}\tInitial Value: {}\tFit: {}\tVary: {}\tMin: {} Max: {}'\
               .format(self.label,
                       self.value,
                       self.fit,
                       self.vary,
                       self.min,
                       self.max)
