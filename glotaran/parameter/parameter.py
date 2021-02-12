"""The parameter class."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from glotaran.parameter import ParameterGroup

import asteval
import numpy as np

RESERVED_LABELS = [symbol for symbol in asteval.make_symbol_table()] + ["group"]


class Keys:
    """Keys for parameter options."""

    EXPR = "expr"
    MAX = "max"
    MIN = "min"
    NON_NEG = "non-negative"
    VARY = "vary"


class Parameter:
    """A parameter for optimization."""

    _find_parameter = re.compile(r"(\$[\w\d\.]+)")
    """A regexpression to find and replace parameter names in expressions."""
    _label_validator_regexp = re.compile(r"\W", flags=re.ASCII)
    """A regexpression to validate labels."""

    def __init__(
        self,
        label: str = None,
        full_label: str = None,
        expression: str = None,
        maximum: int | float = np.inf,
        minimum: int | float = -np.inf,
        non_negative: bool = False,
        value: float = None,
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

        self.label = label
        self.full_label = full_label
        self.expression = expression
        self.maximum = maximum
        self.minimum = minimum
        self.non_negative = non_negative
        self.standard_error = 0.0
        self.value = value
        self.vary = vary

        self._transformed_expression = None

    @classmethod
    def valid_label(cls, label: str) -> bool:
        """Returns true if the `label` is valid string."""
        return cls._label_validator_regexp.search(label) is None and label not in RESERVED_LABELS

    @classmethod
    def from_list_or_value(
        cls,
        value: int | float | list,
        default_options: dict = None,
        label: str = None,
    ) -> Parameter:
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
        options = None

        if not isinstance(value, list):
            param.value = value

        else:
            values = _sanatize_parameter_list(value)
            param.label = _retrieve_from_list_by_type(values, str, label)
            param.value = float(_retrieve_from_list_by_type(values, (int, float), 0))
            options = _retrieve_from_list_by_type(values, dict, None)

        if default_options:
            param._set_options_from_dict(default_options)

        if options:
            param._set_options_from_dict(options)
        return param

    def set_from_group(self, group: ParameterGroup):
        """Sets all values of the parameter to the values of the corresponding parameter in the group.

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
        self.standard_error = p.standard_error
        self.value = p.value
        self.vary = p.vary

    def _set_options_from_dict(self, options: dict):
        if Keys.EXPR in options:
            self.expression = options[Keys.EXPR]
        if Keys.NON_NEG in options:
            self.non_negative = options[Keys.NON_NEG]
        if Keys.MAX in options:
            self.maximum = options[Keys.MAX]
        if Keys.MIN in options:
            self.minimum = options[Keys.MIN]
        if Keys.VARY in options:
            self.vary = options[Keys.VARY]

    @property
    def label(self) -> str:
        """Label of the parameter"""
        return self._label

    @label.setter
    def label(self, label: str):
        if label is not None and not Parameter.valid_label(label):
            raise ValueError("'{label}' is not a valid group label.")
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

        Always `False` if `expression` is not `None`.
        """  # w605
        return self._non_negative if self.expression is None else False

    @non_negative.setter
    def non_negative(self, non_negative: bool):
        self._non_negative = non_negative

    @property
    def vary(self) -> bool:
        """Indicates if the parameter should be optimized.

        Always `False` if `expression` is not `None`.
        """
        return self._vary if self.expression is None else False

    @vary.setter
    def vary(self, vary: bool):
        self._vary = vary

    @property
    def maximum(self) -> float:
        """The upper bound of the parameter."""
        return self._maximum

    @maximum.setter
    def maximum(self, maximum: int | float):
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
    def minimum(self, minimum: int | float):
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
        self._transformed_expression = None

    @property
    def transformed_expression(self) -> str:
        """The expression of the parameter transformed for evaluation within a `ParameterGroup`."""
        if self.expression is not None and self._transformed_expression is None:
            self._transformed_expression = self.expression
            for match in Parameter._find_parameter.findall(self._transformed_expression):
                self._transformed_expression = self._transformed_expression.replace(
                    match, f"group.get('{match[1:]}').value"
                )
        return self._transformed_expression

    @property
    def standard_error(self) -> float:
        """The standard error of the optimized parameter."""
        return self._stderr

    @standard_error.setter
    def standard_error(self, standard_error: float):
        self._stderr = standard_error

    @property
    def value(self) -> float:
        """The value of the parameter"""
        return self._getval()

    @value.setter
    def value(self, value: int | float):
        if not isinstance(value, float) and value is not None:
            try:
                value = float(value)
            except Exception:
                raise TypeError(
                    "Parameter value must be numeric."
                    + f"'{self.full_label}' has value '{value}' of type '{type(value)}'"
                )

        self._value = value

    def get_value_and_bounds_for_optimization(self) -> tuple[float, float, float]:
        """Gets the parameter value and bounds with expression and non-negative constraints
        applied."""
        value = self.value
        minimum = self.minimum
        maximum = self.maximum

        if self.non_negative:
            value = _log_value(value)
            minimum = _log_value(minimum)
            maximum = _log_value(maximum)

        return value, minimum, maximum

    def set_value_from_optimization(self, value: float):
        """Sets the value from an optimization result and reverses non-negative transformation."""
        self.value = np.exp(value) if self.non_negative else value

    def __getstate__(self):
        """Get state for pickle."""
        return (
            self.label,
            self.full_label,
            self.expression,
            self.maximum,
            self.minimum,
            self.non_negative,
            self.standard_error,
            self.value,
            self.vary,
        )

    def __setstate__(self, state):
        """Set state from pickle."""
        (
            self.label,
            self.full_label,
            self.expression,
            self.maximum,
            self.minimum,
            self.non_negative,
            self.standard_error,
            self.value,
            self.vary,
        ) = state

    def _getval(self) -> float:
        return self._value

    def __repr__(self):
        """String representation """
        return (
            f"__{self.label}__: _Value_: {self.value}, _StdErr_: {self.standard_error}, _Min_:"
            f" {self.minimum}, _Max_: {self.maximum}, _Vary_: {self.vary},"
            f" _Non-Negative_: {self.non_negative}, _Expr_: {self.expression}"
        )

    def __array__(self):
        """array"""
        return np.array(float(self._getval()), dtype=float)

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


def _log_value(value: float):
    if not np.isfinite(value):
        return value
    if value == 1:
        value += 1e-10
    return np.log(value)


# A reexp for ONLY matching scientific
_match_scientific = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)")


def _sanatize_parameter_list(li: list) -> list:
    for i, value in enumerate(li):
        if isinstance(value, str) and _match_scientific.match(value):
            li[i] = float(value)

    return li


def _retrieve_from_list_by_type(li: list, t: type, default: Any):
    tmp = list(filter(lambda x: isinstance(x, t), li))
    if not tmp:
        return default
    li.remove(tmp[0])
    return tmp[0]
