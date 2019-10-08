"""The parameter class."""

import typing
import numpy as np

from lmfit import Parameter as LmParameter

import glotaran


class Keys:
    """Keys for parameter options."""
    MIN = "min"
    MAX = "max"
    NON_NEG = "non-negative"
    EXPR = "expr"
    VARY = "vary"


class Parameter(LmParameter):
    """A parameter for optimization."""

    def __init__(self, label: str = None, full_label: str = None):
        """

        Parameters
        ----------
        label :
            The label of the parameter.
        full_label : str
            The label of the parameter with its path in a paramter group prepended.
        """

        super(Parameter, self).__init__(
            user_data={'non_neg': False, 'full_label': full_label})

        self.label = label
        self.full_label = full_label

    @classmethod
    def from_parameter(cls, label: str, parameter: LmParameter) -> 'Parameter':
        """Creates a :class:`Parameter` from a `lmfit.Parameter`

        Parameters
        ----------
        label : str
            The label of the parameter.
        parameter : lmfit.Parameter
            The `lmfit.Parameter`.
        """
        p = cls(label=label)
        p.vary = parameter.vary
        p.non_neg = parameter.user_data['non_neg']
        p.full_label = parameter.user_data['full_label']
        p.value = np.exp(parameter.value) if p.non_neg else parameter.value
        p.min = \
            np.exp(parameter.min) if p.non_neg and np.isfinite(parameter.min) else parameter.min
        p.max = \
            np.exp(parameter.max) if p.non_neg and np.isfinite(parameter.max) else parameter.max
        p.stderr = parameter.stderr
        return p

    @classmethod
    def from_list_or_value(cls,
                           value: typing.Union[int, float, typing.List],
                           default_options: typing.Dict = None,
                           label: str = None) -> 'Parameter':
        """Creates a parameter from a list or numeric value.

        Parameters
        ----------
        value :
            The list or numeric value.
        default_options :
            A dictonary of default options.
        label :
            The label of the parameter.
        """

        if not isinstance(value, list):
            value = [value]

        param = cls(label=label)

        if isinstance(value, (int, float)):
            param.value = value
            return param

        def retrieve(filt, default):
            tmp = list(filter(filt, value))
            if len(tmp) != 0:
                value.remove(tmp[0])
                return tmp[0]
            else:
                return default
        options = retrieve(lambda x: isinstance(x, dict), None)

        param.label = value[0] if len(value) != 1 else label
        param.value = float(value[0] if len(value) == 1 else value[1])

        if default_options:
            param._set_options_from_dict(default_options)

        if options:
            param._set_options_from_dict(options)
        return param

    def set_from_group(self, group: 'glotaran.parameter.ParameterGroup'):
        """Sets all values of the parameter to the values of the conrresoping parameter in the group.

        Notes
        -----

        For internal use.

        Parameters
        ----------
        group :
            The :class:`glotaran.parameter.ParameterGroup`.
        """

        p = group.get(self.full_label)
        self.vary = p.vary
        self.value = p.value
        self.min = p.min
        self.max = p.max
        self.expr = p.expr
        self.stderr = p.stderr
        self.non_neg = p.non_neg

    def _set_options_from_dict(self, options: typing.Dict):
        if Keys.NON_NEG in options:
            self.non_neg = options[Keys.NON_NEG]
        if Keys.MAX in options:
            self.max = float(options[Keys.MAX])
        if Keys.MIN in options:
            self.min = float(options[Keys.MIN])
        if Keys.EXPR in options:
            self.expr = options[Keys.EXPR]
        if Keys.VARY in options:
            self.vary = options[Keys.VARY]

    @property
    def label(self) -> str:
        """Label of the parameter"""
        return self.user_data['label']

    @label.setter
    def label(self, label: str):
        self.user_data['label'] = label

    @property
    def full_label(self) -> str:
        """The label of the parameter with its path in a parameter group prepended."""
        return self.user_data['full_label']

    @full_label.setter
    def full_label(self, full_label: str):
        self.user_data['full_label'] = full_label

    @property
    def non_neg(self) -> bool:
        """Indicates if the parameter is non-negativ.

        If true, the parameter will be transformed with :math:`p' = \log{p}` and
        :math:`p = \exp{p'}`.
        """  # noqa w605
        return self.user_data['non_neg']

    @non_neg.setter
    def non_neg(self, non_neg: bool):
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

        LmParameter.value.fset(self, val)

    def __str__(self):
        """ """
        return f"__{self.label}__: _Value_: {self.value}, _StdErr_: {self.stderr}, _Min_:" + \
               f" {self.min}, _Max_: {self.max}, _Vary_: {self.vary}," + \
               f" _Non-Negative_: {self.non_neg}"
