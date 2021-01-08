"""The parameter class."""

from typing import Dict
from typing import List
from typing import Union

import numpy as np

import glotaran


class Keys:
    """Keys for parameter options."""

    EXPR = "expr"
    MAX = "max"
    MIN = "min"
    NON_NEG = "non-negative"
    VARY = "vary"


class Parameter:
    """A parameter for optimization."""

    def __init__(
        self,
        label: str = None,
        full_label: str = None,
        expression: str = None,
        maximum: Union[int, float] = np.inf,
        minimum: Union[int, float] = -np.inf,
        non_negative: bool = False,
        vary: bool = True,
    ):
        """

        Parameters
        ----------
        label :
            The label of the parameter.
        full_label : str
            The label of the parameter with its path in a parameter group prepended.
        """  # TODO: update docstring.

        super().__init__(name=label, user_data={"non_negative": False, "full_label": full_label})

        self.label = label
        self.full_label = full_label
        self.expression = expression
        self.maximum = maximum
        self.minimum = minimum
        self.non_negative = non_negative
        self.stderr = 0.0
        self.vary = vary

    #  @classmethod
    #  def from_parameter(cls, label: str, parameter: LmParameter) -> "Parameter":
    #      """Creates a :class:`Parameter` from a `lmfit.Parameter`
    #
    #      Parameters
    #      ----------
    #      label : str
    #          The label of the parameter.
    #      parameter : lmfit.Parameter
    #          The `lmfit.Parameter`.
    #      """
    #      p = cls(label=label)
    #      p.vary = parameter.vary
    #      p.non_neg = parameter.user_data["non_neg"]
    #      p.full_label = parameter.user_data["full_label"]
    #      p.value = np.exp(parameter.value) if p.non_neg else parameter.value
    #      p.min = (
    #          np.exp(parameter.min) if p.non_neg and np.isfinite(parameter.min) else parameter.min
    #      )
    #      p.max = (
    #          np.exp(parameter.max) if p.non_neg and np.isfinite(parameter.max) else parameter.max
    #      )
    #      p.stderr = parameter.stderr
    #      return p

    @classmethod
    def from_list_or_value(
        cls,
        value: Union[int, float, List],
        default_options: Dict = None,
        label: str = None,
    ) -> "Parameter":
        """Creates a parameter from a list or numeric value.

        Parameters
        ----------
        value :
            The list or numeric value.
        default_options :
            A dictionary of default options.
        label :
            The label of the parameter.
        """

        param = cls(label=label)

        if not isinstance(value, list):
            param.value = value
            return param

        def retrieve(filt, default):
            tmp = list(filter(filt, value))
            if not tmp:
                return default

            value.remove(tmp[0])
            return tmp[0]

        options = retrieve(lambda x: isinstance(x, dict), None)

        param.label = value[0] if len(value) != 1 else label
        param.value = float(value[0] if len(value) == 1 else value[1])

        if default_options:
            param._set_options_from_dict(default_options)

        if options:
            param._set_options_from_dict(options)
        return param

    def set_from_group(self, group: "glotaran.parameter.ParameterGroup"):
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
        self.expression = p.expression
        self.maximum = p.maximum
        self.minimum = p.minimum
        self.non_negative = p.non_negative
        self.stderr = p.stderr
        self.value = p.value
        self.vary = p.vary

    def _set_options_from_dict(self, options: Dict):
        self.expr = options.get(Keys.EXPR, None)
        self.non_negative = options.get(Keys.NON_NEG, False)
        self.maximum = options.get(Keys.MAX, np.inf)
        self.minimum = options.get(Keys.MIN, -np.inf)
        self.vary = options.get(Keys.VARY, True)

    @property
    def label(self) -> str:
        """Label of the parameter"""
        return self._label

    @label.setter
    def label(self, label: str):
        self._label = label

    @property
    def full_label(self) -> str:
        """The label of the parameter with its path in a parameter group prepended."""
        return self._full_label

    @full_label.setter
    def full_label(self, full_label: str):
        self._full_label = full_label

    @property
    def non_negative(self) -> bool:
        r"""Indicates if the parameter is non-negativ.

        If true, the parameter will be transformed with :math:`p' = \log{p}` and
        :math:`p = \exp{p'}`.
        """  # w605
        return self._non_negative

    @non_negative.setter
    def non_negative(self, non_negative: bool):
        self._non_negative = non_negative

    @property
    def vary(self) -> bool:
        """Indicates if the parameter should be optimized."""
        return self._vary

    @vary.setter
    def vary(self, vary: bool):
        self._vary = vary

    @property
    def maximum(self) -> float:
        """The upper bound of the parameter."""
        return self._maximum

    @maximum.setter
    def maximum(self, maximum: Union[int, float]):
        if not isinstance(maximum, float):
            try:
                maximum = float(maximum)
            except Exception:
                raise TypeError(
                    "Parameter maximum must be numeric."
                    + f"'{self.full_label}' has maximum '{maximum}' of type '{type(maximum)}'"
                )

        self._maximum = maximum

    @property
    def minimum(self) -> float:
        """The lower bound of the parameter."""
        return self._minimum

    @minimum.setter
    def minimum(self, minimum: Union[int, float]):
        if not isinstance(minimum, float):
            try:
                minimum = float(minimum)
            except Exception:
                raise TypeError(
                    "Parameter minimum must be numeric."
                    + f"'{self.full_label}' has minimum '{minimum}' of type '{type(minimum)}'"
                )

        self._minimum = minimum

    @property
    def expression(self) -> str:
        """The expression of the parameter."""  # TODO: Formulate better docstring.
        return self._expression

    @expression.setter
    def expression(self, expression: str):
        self._expression = expression

    @property
    def stderr(self) -> float:
        """The standard error of the optimized parameter."""
        return self._stderr

    @stderr.setter
    def stderr(self, stderr: float):
        self._stderr = stderr

    @property
    def value(self) -> float:
        """The value of the parameter"""
        return self._getval()

    @value.setter
    def value(self, value: Union[int, float]):
        if not isinstance(value, float):
            try:
                value = float(value)
            except Exception:
                raise TypeError(
                    "Parameter value must be numeric."
                    + f"'{self.full_label}' has value '{value}' of type '{type(value)}'"
                )

        self._value = value

    def _getval(self) -> float:
        return self._value

    def __repr__(self):
        """String representation """
        return (
            f"__{self.label}__: _Value_: {self.value}, _StdErr_: {self.stderr}, _Min_:"
            + f" {self.min}, _Max_: {self.max}, _Vary_: {self.vary},"
            + f" _Non-Negative_: {self.non_negative}"
        )

    def __array__(self):
        """array"""
        return [float(self._getval())]

    def __str__(self):
        """string"""
        return self.__repr__()

    def __abs__(self):
        """abs"""
        return abs(self._getval())

    def __neg__(self):
        """neg"""
        return -self._getval()

    def __pos__(self):
        """positive"""
        return +self._getval()

    def __bool__(self):
        """bool"""
        return self._getval() != 0

    def __int__(self):
        """int"""
        return int(self._getval())

    def __float__(self):
        """float"""
        return float(self._getval())

    def __trunc__(self):
        """trunc"""
        return self._getval().__trunc__()

    def __add__(self, other):
        """+"""
        return self._getval() + other

    def __sub__(self, other):
        """-"""
        return self._getval() - other

    def __truediv__(self, other):
        """/"""
        return self._getval() / other

    def __floordiv__(self, other):
        """//"""
        return self._getval() // other

    def __divmod__(self, other):
        """divmod"""
        return divmod(self._getval(), other)

    def __mod__(self, other):
        """%"""
        return self._getval() % other

    def __mul__(self, other):
        """*"""
        return self._getval() * other

    def __pow__(self, other):
        """**"""
        return self._getval() ** other

    def __gt__(self, other):
        """>"""
        return self._getval() > other

    def __ge__(self, other):
        """>="""
        return self._getval() >= other

    def __le__(self, other):
        """<="""
        return self._getval() <= other

    def __lt__(self, other):
        """<"""
        return self._getval() < other

    def __eq__(self, other):
        """=="""
        return self._getval() == other

    def __ne__(self, other):
        """!="""
        return self._getval() != other

    def __radd__(self, other):
        """+ (right)"""
        return other + self._getval()

    def __rtruediv__(self, other):
        """/ (right)"""
        return other / self._getval()

    def __rdivmod__(self, other):
        """divmod (right)"""
        return divmod(other, self._getval())

    def __rfloordiv__(self, other):
        """// (right)"""
        return other // self._getval()

    def __rmod__(self, other):
        """% (right)"""
        return other % self._getval()

    def __rmul__(self, other):
        """* (right)"""
        return other * self._getval()

    def __rpow__(self, other):
        """** (right)"""
        return other ** self._getval()

    def __rsub__(self, other):
        """- (right)"""
        return other - self._getval()
