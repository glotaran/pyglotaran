""" Glotaran Parameter"""

from typing import List, Union
from math import isnan

from lmfit import Parameter as LmParameter


class Keys:
    MIN = "min"
    MAX = "max"
    EXPR = "expr"
    VARY = "vary"


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

    @classmethod
    def from_list_or_value(cls, parameter: Union[int, float, List[object]]):
            param = cls()

            if isinstance(parameter, (int, float)):
                param.value = parameter
                return param

            def retrieve(filt, default):
                tmp = list(filter(filt, parameter))
                return tmp[0] if tmp else default

            def filt_num(x):
                return isinstance(x, (int, float)) and not isinstance(x, bool)

            param.value = retrieve(filt_num, 'nan')
            param.label = retrieve(lambda x: isinstance(x, str) and not
                                   x == 'nan', None)
            param.fit = retrieve(lambda x: isinstance(x, bool), True)
            options = retrieve(lambda x: isinstance(x, dict), None)

            if options is not None:
                if Keys.MAX in options:
                    param.max = options[Keys.MAX]
                if Keys.MIN in options:
                    param.min = options[Keys.MIN]
                if Keys.EXPR in options:
                    param.expr = options[Keys.EXPR]
                if Keys.VARY in options:
                    param.vary = options[Keys.VARY]
            return param

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
