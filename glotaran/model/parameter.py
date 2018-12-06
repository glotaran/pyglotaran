"""This package contains glotarans parameter class"""

from typing import List, Union
from math import isnan, exp

from lmfit import Parameter as LmParameter


class Keys:
    MIN = "min"
    MAX = "max"
    NON_NEG = "non-negative"
    EXPR = "expr"
    VARY = "vary"


class Parameter(LmParameter):
    """Wrapper for lmfit parameter."""
    def __init__(self, label=None):
        self.index = -1
        self._non_neg = True
        self.label = label
        super(Parameter, self).__init__(user_data={'non_neg': True})

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
        p = cls(label=label)
        p.min = parameter.min
        p.max = parameter.max
        p.vary = parameter.vary
        p.non_neg = parameter.user_data['non_neg']
        p.value = exp(parameter.value) if p.non_neg else parameter.value
        return p

    @classmethod
    def from_list_or_value(cls, parameter: Union[int, float, List[object]],
                           label=None):

        if not isinstance(parameter, list):
            parameter = [parameter]

        param = cls(label=label)

        if isinstance(parameter, (int, float)):
            param.value = parameter
            return param

        def retrieve(filt, default):
            tmp = list(filter(filt, parameter))
            if len(tmp) is not 0:
                parameter.remove(tmp[0])
                return tmp[0]
            else:
                return default
        options = retrieve(lambda x: isinstance(x, dict), None)

        param.label = parameter[0] if len(parameter) is not 1 else label
        param.value = float(parameter[0] if len(parameter) is 1 else parameter[1])

        if options is not None:
            if Keys.NON_NEG in options:
                param.non_neg = options[Keys.NON_NEG]
            if Keys.MAX in options:
                param.max = float(options[Keys.MAX])
            if Keys.MIN in options:
                param.min = float(options[Keys.MIN])
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
    def non_neg(self) -> bool:
        return self._non_neg

    @non_neg.setter
    def non_neg(self, non_neg: bool):
        self._non_neg = non_neg
        self.user_data['non_neg'] = non_neg

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
               f" {self.min}, _Max_: {self.max}, _Vary_: {self.vary}," + \
               f"_Non-Negative_: {self.non_neg}"
