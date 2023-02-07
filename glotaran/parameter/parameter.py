"""The parameter class."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any

import asteval
import numpy as np
from attr import ib
from attrs import Attribute
from attrs import asdict
from attrs import define
from attrs import evolve
from attrs import fields
from attrs import filters
from attrs import validators

from glotaran.typing.types import _SupportsArray
from glotaran.utils.attrs_helper import no_default_vals_in_repr
from glotaran.utils.helpers import nan_or_equal
from glotaran.utils.ipython import MarkdownStr
from glotaran.utils.sanitize import pretty_format_numerical
from glotaran.utils.sanitize import sanitize_parameter_list

if TYPE_CHECKING:
    from glotaran.parameter import Parameters

RESERVED_LABELS: list[str] = list(asteval.make_symbol_table().keys()) + ["parameters", "iteration"]


OPTION_NAMES_SERIALIZED = {
    "expression": "expr",
    "maximum": "max",
    "minimum": "min",
    "non_negative": "non-negative",
    "standard_error": "standard-error",
}

OPTION_NAMES_DESERIALIZED = {v: k for k, v in OPTION_NAMES_SERIALIZED.items()}


def deserialize_options(options: dict[str, Any]) -> dict[str, Any]:
    """Replace options keys in serialized format by attribute names.

    Parameters
    ----------
    options : dict[str, Any]
        The serialized options.

    Returns
    -------
    dict[str, Any]
        The deserialized options.

    """
    return {OPTION_NAMES_DESERIALIZED.get(k, k): v for k, v in options.items()}


def serialize_options(options: dict[str, Any]) -> dict[str, Any]:
    """Replace options keys with serialized format by attribute names.

    Parameters
    ----------
    options : dict[str, Any]
        The options.

    Returns
    -------
    dict[str, Any]
        The serialized options.

    """
    return {OPTION_NAMES_SERIALIZED.get(k, k): v for k, v in options.items()}


PARAMETER_EXPRESSION_REGEX = re.compile(r"\$(?P<parameter_expression>[\w\d\.]+)((?![\w\d\.]+)|$)")
"""A regular expression to find and replace parameter names in expressions."""
VALID_LABEL_REGEX = re.compile(r"\W", flags=re.ASCII)
"""A regular expression to validate labels."""


def valid_label(parameter: Parameter, attribute: Attribute, label: str):
    """Check if a label is a valid label for :class:`Parameter`.

    Parameters
    ----------
    parameter : Parameter
        The :class:`Parameter` instance
    attribute : Attribute
        The label field.
    label : str
        The label value.

    Raises
    ------
    ValueError
        Raise when the label is not valid.
    """
    if VALID_LABEL_REGEX.search(label.replace(".", "_")) is not None or label in RESERVED_LABELS:
        raise ValueError(f"'{label}' is not a valid parameter label.")


def set_transformed_expression(parameter: Parameter, attribute: Attribute, expression: str | None):
    """Set the transformed expression from an expression.

    Parameters
    ----------
    parameter : Parameter
        The :class:`Parameter` instance
    attribute : Attribute
        The label field.
    expression : str | None
        The expression value.
    """
    if expression:
        parameter.vary = False
        parameter.transformed_expression = PARAMETER_EXPRESSION_REGEX.sub(
            r"parameters.get('\g<parameter_expression>').value", expression
        )


@no_default_vals_in_repr
@define
class Parameter(_SupportsArray):
    """A parameter for optimization."""

    label: str = ib(converter=str, validator=[valid_label])
    value: float = ib(
        default=np.nan,
        converter=lambda v: float(v) if isinstance(v, int) else v,
        validator=[validators.instance_of(float)],
    )
    standard_error: float = np.nan
    expression: str | None = ib(default=None, validator=[set_transformed_expression])
    maximum: float = ib(default=np.inf, validator=[validators.instance_of((int, float))])
    minimum: float = ib(default=-np.inf, validator=[validators.instance_of((int, float))])
    non_negative: bool = False
    vary: bool = ib(default=True)

    transformed_expression: str | None = ib(default=None, init=False, repr=False)

    @property
    def label_short(self) -> str:
        """Get short label.

        Returns
        -------
        str :
            The short label.
        """
        return self.label.split(".")[-1]

    @classmethod
    def from_list(
        cls,
        values: list[Any],
        *,
        default_options: dict[str, Any] | None = None,
    ) -> Parameter:
        """Create a parameter from a list.

        Parameters
        ----------
        values : list[Any]
            The list of parameter definitions.
        default_options : dict[str, Any] | None
            A dictionary of default options.

        Returns
        -------
        Parameter
            The created :class:`Parameter`.
        """
        options = None

        values = sanitize_parameter_list(values.copy())
        param = {
            "label": _retrieve_item_from_list_by_type(values, str, ""),
            "value": _retrieve_item_from_list_by_type(values, (int, float), np.nan),
        }
        options = _retrieve_item_from_list_by_type(values, dict, {})

        if default_options:
            param |= deserialize_options(default_options)
        param |= deserialize_options(options)

        return cls(**param)

    def copy(self) -> Parameter:
        """Create a copy of the :class:`Parameter`.

        Returns
        -------
        Parameter :
            A copy of the :class:`Parameter`.
        """
        return evolve(self)

    def as_dict(self) -> dict[str, Any]:
        """Get the parameter as a dictionary.

        Returns
        -------
        dict[str, Any]
            The parameter as dictionary.
        """
        return asdict(self, filter=filters.exclude(fields(Parameter).transformed_expression))

    def _deep_equals(self, other: Parameter) -> bool:
        """Compare all attributes for equality not only ``value`` like ``__eq__`` does.

        This is used by ``Parameters`` to check for equality.

        Parameters
        ----------
        other: Parameter
            Other parameter to compare against.

        Returns
        -------
        bool
            Whether or not all attributes are equal.
        """
        return all(
            nan_or_equal(self_val, other_val)
            for self_val, other_val in zip(self.as_dict().values(), other.as_dict().values())
        )

    def as_list(self, label_short: bool = False) -> list[str | float | dict[str, Any]]:
        """Get the parameter as a dictionary.

        Parameters
        ----------
        label_short : bool
            If true, the label will be replaced by the shortened label.

        Returns
        -------
        dict[str, Any]
            The parameter as dictionary.
        """
        options = self.as_dict()

        label = options.pop("label")
        value = options.pop("value")

        if label_short:
            label = self.label_short

        return [label, value, serialize_options(options)]

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

    def markdown(
        self,
        all_parameters: Parameters | None = None,
        initial_parameters: Parameters | None = None,
    ) -> MarkdownStr:
        """Get a markdown representation of the parameter.

        Parameters
        ----------
        all_parameters : Parameters | None
            A parameter group containing the whole parameter set (used for expression lookup).
        initial_parameters : Parameters | None
            The initial parameter.

        Returns
        -------
        MarkdownStr
            The parameter as markdown string.
        """
        md = f"{self.label}"

        parameter = self if all_parameters is None else all_parameters.get(self.label)
        value = f"{parameter.value:.2e}"
        if parameter.vary:
            if parameter.standard_error is not np.nan:
                t_value = pretty_format_numerical(parameter.value / parameter.standard_error)
                value += f"±{parameter.standard_error:.2e}, t-value: {t_value}"

            if initial_parameters is not None:
                initial_value = initial_parameters.get(parameter.label).value
                value += f", initial: {initial_value:.2e}"
            md += f"({value})"
        elif parameter.expression is not None:
            expression = parameter.expression
            if all_parameters is not None:
                for match in PARAMETER_EXPRESSION_REGEX.findall(expression):
                    label = match[0]
                    parameter = all_parameters.get(label)
                    expression = expression.replace(
                        f"${label}", f"_{parameter.markdown(all_parameters=all_parameters)}_"
                    )

            md += f"({value}={expression})"
        else:
            md += f"({value}, fixed)"

        return MarkdownStr(md)

    def __array__(self):
        """array"""  # noqa: D400, D403
        return np.array(self.value, dtype=float)

    def __str__(self) -> str:
        """Representation used by print and str."""
        return (
            f"__{self.label}__: _Value_: {self.value}, _StdErr_: {self.standard_error}, _Min_:"
            f" {self.minimum}, _Max_: {self.maximum}, _Vary_: {self.vary},"
            f" _Non-Negative_: {self.non_negative}, _Expr_: {self.expression}"
        )

    def __abs__(self):
        """abs"""  # noqa: D400, D403
        return abs(self.value)

    def __neg__(self):
        """neg"""  # noqa: D400, D403
        return -self.value

    def __pos__(self):
        """positive"""  # noqa: D400, D403
        return +self.value

    def __int__(self):
        """int"""  # noqa: D400, D403
        return int(self.value)

    def __float__(self):
        """float"""  # noqa: D400, D403
        return float(self.value)

    def __trunc__(self):
        """trunc"""  # noqa: D400, D403
        return self.value.__trunc__()

    def __add__(self, other):
        """+"""  # noqa: D400
        return self.value + other

    def __sub__(self, other):
        """-"""  # noqa: D400
        return self.value - other

    def __truediv__(self, other):
        """/"""  # noqa: D400
        return self.value / other

    def __floordiv__(self, other):
        """//"""  # noqa: D400
        return self.value // other

    def __divmod__(self, other):
        """divmod"""  # noqa: D400, D403
        return divmod(self.value, other)

    def __mod__(self, other):
        """%"""  # noqa: D400
        return self.value % other

    def __mul__(self, other):
        """*"""  # noqa: D400
        return self.value * other

    def __pow__(self, other):
        """**"""  # noqa: D400
        return self.value**other

    def __gt__(self, other):
        """>"""  # noqa: D400
        return self.value > other

    def __ge__(self, other):
        """>="""  # noqa: D400
        return self.value >= other

    def __le__(self, other):
        """<="""  # noqa: D400
        return self.value <= other

    def __lt__(self, other):
        """<"""  # noqa: D400
        return self.value < other

    def __eq__(self, other):
        """=="""  # noqa: D400
        return self.value == other

    def __ne__(self, other):
        """!="""  # noqa: D400
        return self.value != other

    def __radd__(self, other):
        """+ (right)"""  # noqa: D400
        return other + self.value

    def __rtruediv__(self, other):
        """/ (right)"""  # noqa: D400
        return other / self.value

    def __rdivmod__(self, other):
        """divmod (right)"""  # noqa: D400, D403
        return divmod(other, self.value)

    def __rfloordiv__(self, other):
        """// (right)"""  # noqa: D400
        return other // self.value

    def __rmod__(self, other):
        """% (right)"""  # noqa: D400
        return other % self.value

    def __rmul__(self, other):
        """* (right)"""  # noqa: D400
        return other * self.value

    def __rpow__(self, other):
        """** (right)"""  # noqa: D400
        return other**self.value

    def __rsub__(self, other):
        """- (right)"""  # noqa: D400
        return other - self.value


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
