""" Glotaran Parameter"""

from math import isnan

from lmfit import Parameter as LmParameter


class Parameter(LmParameter):
    """Wrapper for lmfit parameter."""
    def __init__(self):
        self.index = -1
        self.fit = True
        self.label = None
        super(Parameter, self).__init__()

    @classmethod
    def from_parameter(cls, label: str, parameter: LmParameter):
        """Creates a parameter from an lmfit.Parameter

        Parameters
        ----------
        label : str
            Label of the parameter
        parameter : lmfit.Parameter
            lmfit.Parameter
        """
        p = cls()
        p.label = label
        p.value = parameter.value
        p.min = parameter.min
        p.max = parameter.max
        p.vary = parameter.vary
        return p

    @property
    def label(self) -> str:
        """Label of the parameter"""
        return self._label

    @label.setter
    def label(self, label: str):
        self._label = label

    @property
    def fit(self):
        """Whether the paramater should be included in fit. Set false for e.g.
        dormant parameter."""
        return self._fit

    @fit.setter
    def fit(self, value):
        if not isinstance(value, bool):
            raise TypeError("Fit must be True or False")
        self._fit = value

    @LmParameter.value.setter
    def value(self, val):
        if not isinstance(val, (int, float)):
                try:
                    val = float(val)
                except Exception:
                    raise Exception("Parameter Error: value must be numeric:"
                                    "{} Type: {}".format(val, type(val)))

        if isinstance(val, int):
            val = float(val)

        if isnan(val):
            self.vary = False

        LmParameter.value.fset(self, val)

    def __str__(self):
        """ """
        return f"__{self.label}__: _Value_: {self.value}, _Min_:" + \
               f" {self.min}, _Max_: {self.max}, _Vary_: {self.vary}, _Fit_: {self.fit}"
