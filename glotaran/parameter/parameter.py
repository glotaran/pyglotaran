"""The parameter class."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import asteval
import numpy as np
from numpy.typing._array_like import _SupportsArray

from glotaran.utils.sanitize import sanitize_parameter_list

if TYPE_CHECKING:
    from typing import Any

    from glotaran.parameter import ParameterGroup

RESERVED_LABELS: list[str] = list(asteval.make_symbol_table().keys()) + ["group"]


class Keys:
    """Keys for parameter options."""

    EXPR = "expr"
    MAX = "max"
    MIN = "min"
    NON_NEG = "non-negative"
    VARY = "vary"


PARAMETER_EXPRESION_REGEX = re.compile(r"\$(?P<parameter_expression>[\w\d\.]+)((?![\w\d\.]+)|$)")
"""A regexpression to find and replace parameter names in expressions."""
VALID_LABEL_REGEX = re.compile(r"\W", flags=re.ASCII)
"""A regexpression to validate labels."""


class Parameter(_SupportsArray):
    """A parameter for optimization."""

    def __init__(
        self,
        label: str = None,
        full_label: str = None,
        expression: str = None,
        maximum: int | float = np.inf,
        minimum: int | float = -np.inf,
        non_negative: bool = False,
        value: float | int = np.nan,
        vary: bool = True,
    ):
        """Optimization Parameter supporting numpy array operations.

        Parameters
        ----------
        label : str
            The label of the parameter., by default None
        full_label : str
            The label of the parameter with its path in a parameter group prepended.
            , by default None
        expression : str
            Expression to calculate the parameters value from,
            e.g. if used in relation to another parameter. , by default None
        maximum : int
            Upper boundary for the parameter to be varied to., by default np.inf
        minimum : int
            Lower boundary for the parameter to be varied to., by default -np.inf
        non_negative : bool
            Whether the parameter should always be bigger than zero., by default False
        value : float
            Value of the parameter, by default np.nan
        vary : bool
            Whether the parameter should be changed during optimization or not.
            , by default True
        """
        self.label = label
        self.full_label = full_label or ""
        self.expression = expression
        self.maximum = maximum
        self.minimum = minimum
        self.non_negative = non_negative
        self.standard_error = 0.0
        self.value = value
        self.vary = vary

        self._transformed_expression: str | None = None

    @staticmethod
    def valid_label(label: str) -> bool:
        """Check if a label is a valid label for :class:`Parameter`.

        Parameters
        ----------
        label : str
            The label to validate.

        Returns
        -------
        bool
            Whether the label is valid.

        """
        return VALID_LABEL_REGEX.search(label) is None and label not in RESERVED_LABELS

    @classmethod
    def from_list_or_value(
        cls,
        value: int | float | list,
        default_options: dict = None,
        label: str = None,
    ) -> Parameter:
        """Create a parameter from a list or numeric value.

        Parameters
        ----------
        value : int | float | list
            The list or numeric value.
        default_options : dict
            A dictionary of default options.
        label : str
            The label of the parameter.

        Returns
        -------
        Parameter
            The created :class:`Parameter`.
        """
        param = cls(label=label)
        options = None

        if not isinstance(value, list):
            param.value = value

        else:
            values = sanitize_parameter_list(value)
            param.label = _retrieve_item_from_list_by_type(values, str, label)
            param.value = float(_retrieve_item_from_list_by_type(values, (int, float), 0))
            options = _retrieve_item_from_list_by_type(values, dict, None)

        if default_options:
            param._set_options_from_dict(default_options)

        if options:
            param._set_options_from_dict(options)
        return param

    def set_from_group(self, group: ParameterGroup):
        """Set all values of the parameter to the values of the corresponding parameter in the group.

        Notes
        -----
        For internal use.

        Parameters
        ----------
        group : ParameterGroup
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
        """Set the parameter's options from a dictionary.

        Parameters
        ----------
        options : dict
            A dictionary containing parameter options.
        """
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
    def label(self) -> str | None:
        """Label of the parameter.

        Returns
        -------
        str
            The label.
        """
        return self._label

    @label.setter
    def label(self, label: str | None):
        # ensure that label is str | None even if an int is passed
        label = None if label is None else str(label)
        if label is not None and not Parameter.valid_label(label):
            raise ValueError(f"'{label}' is not a valid group label.")
        self._label = label

    @property
    def full_label(self) -> str:
        """Label of the parameter with its path in a parameter group prepended.

        Returns
        -------
        str
            The full label.
        """
        return self._full_label

    @full_label.setter
    def full_label(self, full_label: str):
        self._full_label = str(full_label)

    @property
    def non_negative(self) -> bool:
        r"""Indicate if the parameter is non-negativ.

        If true, the parameter will be transformed with :math:`p' = \log{p}` and
        :math:`p = \exp{p'}`.

        Notes
        -----
        Always `False` if `expression` is not `None`.

        Returns
        -------
        bool
            Whether the parameter is non-negativ.
        """
        return self._non_negative if self.expression is None else False

    @non_negative.setter
    def non_negative(self, non_negative: bool):
        self._non_negative = non_negative

    @property
    def vary(self) -> bool:
        """Indicate if the parameter should be optimized.

        Notes
        -----
        Always `False` if `expression` is not `None`.

        Returns
        -------
        bool
            Whether the parameter should be optimized.
        """
        return self._vary if self.expression is None else False

    @vary.setter
    def vary(self, vary: bool):
        self._vary = vary

    @property
    def maximum(self) -> float:
        """Upper bound of the parameter.

        Returns
        -------
        float
            The upper bound of the parameter.
        """
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
        """Lower bound of the parameter.

        Returns
        -------
        float

            The lower bound of the parameter.
        """
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
    def expression(self) -> str | None:
        """Expression to calculate the parameters value from.

        This can used to set a relation to another parameter.

        Returns
        -------
        str | None
            The expression.
        """
        return self._expression

    @expression.setter
    def expression(self, expression: str | None):
        self._expression = expression
        self._transformed_expression = None

    @property
    def transformed_expression(self) -> str | None:
        """Expression of the parameter transformed for evaluation within a `ParameterGroup`.

        Returns
        -------
        str | None
            The transformed expression.
        """
        if self.expression is not None and self._transformed_expression is None:
            self._transformed_expression = PARAMETER_EXPRESION_REGEX.sub(
                r"group.get('\g<parameter_expression>').value", self.expression
            )
        return self._transformed_expression

    @property
    def standard_error(self) -> float:
        """Standard error of the optimized parameter.

        Returns
        -------
        float
            The standard error of the parameter.
        """  # noqa: D401
        return self._stderr

    @standard_error.setter
    def standard_error(self, standard_error: float):
        self._stderr = standard_error

    @property
    def value(self) -> float:
        """Value of the parameter.

        Returns
        -------
        float
            The value of the parameter.
        """
        return self._value

    @value.setter
    def value(self, value: int | float):
        if not isinstance(value, float) and value is not np.nan:
            try:
                value = float(value)
            except Exception:
                raise TypeError(
                    "Parameter value must be numeric."
                    + f"'{self.full_label}' has value '{value}' of type '{type(value)}'"
                )

        self._value = value

    def get_value_and_bounds_for_optimization(self) -> tuple[float, float, float]:
        """Get the parameter value and bounds with expression and non-negative constraints applied.

        Returns
        -------
        tuple[float, float, float]
            A tuple containing the value, the lower and the upper bound.
        """
        value = self.value
        minimum = self.minimum
        maximum = self.maximum

        if self.non_negative:
            value = _log_value(value)
            minimum = _log_value(minimum)
            maximum = _log_value(maximum)

        return value, minimum, maximum

    def set_value_from_optimization(self, value: float):
        """Set the value from an optimization result and reverses non-negative transformation.

        Parameters
        ----------
        value : float
            Value from optimization.
        """
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

    def __repr__(self):
        """Representation used by repl and tracebacks."""
        return (
            f"{type(self).__name__}(label={self.label!r}, value={self.value!r},"
            f" expression={self.expression!r}, vary={self.vary!r})"
        )

    def __array__(self):
        """array"""  # noqa: D400, D403
        return np.array(float(self._value), dtype=float)

    def __str__(self) -> str:
        """Representation used by print and str."""
        return (
            f"__{self.label}__: _Value_: {self.value}, _StdErr_: {self.standard_error}, _Min_:"
            f" {self.minimum}, _Max_: {self.maximum}, _Vary_: {self.vary},"
            f" _Non-Negative_: {self.non_negative}, _Expr_: {self.expression}"
        )

    def __abs__(self):
        """abs"""  # noqa: D400, D403
        return abs(self._value)

    def __neg__(self):
        """neg"""  # noqa: D400, D403
        return -self._value

    def __pos__(self):
        """positive"""  # noqa: D400, D403
        return +self._value

    def __int__(self):
        """int"""  # noqa: D400, D403
        return int(self._value)

    def __float__(self):
        """float"""  # noqa: D400, D403
        return float(self._value)

    def __trunc__(self):
        """trunc"""  # noqa: D400, D403
        return self._value.__trunc__()

    def __add__(self, other):
        """+"""  # noqa: D400
        return self._value + other

    def __sub__(self, other):
        """-"""  # noqa: D400
        return self._value - other

    def __truediv__(self, other):
        """/"""  # noqa: D400
        return self._value / other

    def __floordiv__(self, other):
        """//"""  # noqa: D400
        return self._value // other

    def __divmod__(self, other):
        """divmod"""  # noqa: D400, D403
        return divmod(self._value, other)

    def __mod__(self, other):
        """%"""  # noqa: D400
        return self._value % other

    def __mul__(self, other):
        """*"""  # noqa: D400
        return self._value * other

    def __pow__(self, other):
        """**"""  # noqa: D400
        return self._value ** other

    def __gt__(self, other):
        """>"""  # noqa: D400
        return self._value > other

    def __ge__(self, other):
        """>="""  # noqa: D400
        return self._value >= other

    def __le__(self, other):
        """<="""  # noqa: D400
        return self._value <= other

    def __lt__(self, other):
        """<"""  # noqa: D400
        return self._value < other

    def __eq__(self, other):
        """=="""  # noqa: D400
        return self._value == other

    def __ne__(self, other):
        """!="""  # noqa: D400
        return self._value != other

    def __radd__(self, other):
        """+ (right)"""  # noqa: D400
        return other + self._value

    def __rtruediv__(self, other):
        """/ (right)"""  # noqa: D400
        return other / self._value

    def __rdivmod__(self, other):
        """divmod (right)"""  # noqa: D400, D403
        return divmod(other, self._value)

    def __rfloordiv__(self, other):
        """// (right)"""  # noqa: D400
        return other // self._value

    def __rmod__(self, other):
        """% (right)"""  # noqa: D400
        return other % self._value

    def __rmul__(self, other):
        """* (right)"""  # noqa: D400
        return other * self._value

    def __rpow__(self, other):
        """** (right)"""  # noqa: D400
        return other ** self._value

    def __rsub__(self, other):
        """- (right)"""  # noqa: D400
        return other - self._value


def _log_value(value: float) -> float:
    """Get the logarithm of a value.

    Performs a check for edge cases and migitates numerical issues.

    Parameters
    ----------
    value : float
        The initial value.

    Returns
    -------
    float
        The logarithm of the value.
    """
    if not np.isfinite(value):
        return value
    if value == 1:
        value += 1e-10
    return np.log(value)


def _retrieve_item_from_list_by_type(
    item_list: list, item_type: type | tuple[type, ...], default: Any
) -> Any:
    """Retrieve an item from list which matches a given type.

    Parameters
    ----------
    item_list : list
        The list to retrieve from.
    item_type : type | tuple[type, ...]
        The item type or tuple of types to match.
    default : Any
        Returned if no item matches.

    Returns
    -------
    Any

    """
    tmp = list(filter(lambda x: isinstance(x, item_type), item_list))
    if not tmp:
        return default
    item_list.remove(tmp[0])
    return tmp[0]
