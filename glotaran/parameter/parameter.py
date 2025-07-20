"""The parameter class."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any

import asteval
import numpy as np
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from pydantic.functional_validators import BeforeValidator

from glotaran.utils.helpers import nan_or_equal
from glotaran.utils.ipython import MarkdownStr
from glotaran.utils.sanitize import pretty_format_numerical
from glotaran.utils.sanitize import sanitize_parameter_list

if TYPE_CHECKING:
    from pydantic._internal import _repr

    from glotaran.parameter import Parameters

RESERVED_LABELS: tuple[str] = (
    *tuple(asteval.make_symbol_table().keys()),
    "parameters",
    "iteration",
)

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


def validate_label(label: Any) -> str:  # noqa: ANN401
    """Check if a label is a valid label for :class:`Parameter`.

    Parameters
    ----------
    label : Any
        The label value passed as argument to :class:`Parameter`.

    Raises
    ------
    ValueError
        Raise when the label is not valid.
    """
    label = str(label)
    if VALID_LABEL_REGEX.search(label.replace(".", "_")) is not None or label in RESERVED_LABELS:
        msg = f"'{label}' is not a valid parameter label."
        raise ValueError(msg)
    return label


class Parameter(BaseModel):
    """A parameter for optimization."""

    label: Annotated[str, BeforeValidator(validate_label)]
    value: Annotated[float, Field(default=np.nan)]
    standard_error: Annotated[float, Field(default=np.nan)]
    expression: Annotated[str | None, Field(default=None)]
    maximum: Annotated[float, Field(default=np.inf)]
    minimum: Annotated[float, Field(default=-np.inf)]
    non_negative: Annotated[bool, Field(default=False)]
    vary: Annotated[bool, Field(default=True)]

    _transformed_expression: Annotated[str | None, Field(default=None, init_var=False, repr=False)]

    @model_validator(mode="after")
    def _set_transformed_expression(self) -> Parameter:
        """Set ``_transformed_expression`` after instance was initialized."""
        if self.expression:
            self.vary = False
            self._transformed_expression = PARAMETER_EXPRESSION_REGEX.sub(
                r"parameters.get('\g<parameter_expression>').value", self.expression
            )
        return self

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

    def get_dependency_parameters(self) -> list[str]:
        return (
            [
                match[0].replace("$", "")
                for match in PARAMETER_EXPRESSION_REGEX.finditer(self.expression)
            ]
            if self.expression is not None
            else []
        )

    def as_dict(self) -> dict[str, Any]:
        """Get the parameter as a dictionary.

        Returns
        -------
        dict[str, Any]
            The parameter as dictionary.
        """
        # return asdict(self, filter=filters.exclude(fields(Parameter).transformed_expression))
        return self.model_dump(exclude={"transformed_expression"})

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
            for self_val, other_val in zip(
                self.as_dict().values(), other.as_dict().values(), strict=False
            )
        )

    def as_list(self, *, label_short: bool = False) -> list[str | float | dict[str, Any]]:
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

    def set_value_from_optimization(self, value: float) -> None:
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
            if np.isnan(parameter.standard_error) is False:
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
                        f"${label}",
                        f"_{parameter.markdown(all_parameters=all_parameters)}_",
                    )

            md += f"({value}={expression})"
        else:
            md += f"({value}, fixed)"

        return MarkdownStr(md)

    def __array__(self) -> np.ndarray:
        """array"""  # noqa: D400, D403
        return np.array(self.value, dtype=float)

    def __str__(self) -> str:
        """Representation used by print and str."""
        return (
            f"__{self.label}__: _Value_: {self.value}, _StdErr_: {self.standard_error}, _Min_:"
            f" {self.minimum}, _Max_: {self.maximum}, _Vary_: {self.vary},"
            f" _Non-Negative_: {self.non_negative}, _Expr_: {self.expression}"
        )

    def __abs__(self) -> float:
        """abs"""  # noqa: D400, D403
        return abs(self.value)

    def __neg__(self) -> float:
        """neg"""  # noqa: D400, D403
        return -self.value

    def __pos__(self) -> float:
        """positive"""  # noqa: D400, D403
        return +self.value

    def __int__(self) -> int:
        """int"""  # noqa: D400, D403
        return int(self.value)

    def __float__(self) -> float:
        """float"""  # noqa: D400, D403
        return float(self.value)

    def __trunc__(self) -> int:
        """trunc"""  # noqa: D400, D403
        return self.value.__trunc__()

    def __add__(self, other):  # noqa: ANN001, ANN204
        """+"""  # noqa: D400
        return self.value + other

    def __sub__(self, other):  # noqa: ANN001, ANN204
        """-"""  # noqa: D400
        return self.value - other

    def __truediv__(self, other):  # noqa: ANN001, ANN204
        """/"""  # noqa: D400
        return self.value / other

    def __floordiv__(self, other):  # noqa: ANN001, ANN204
        """//"""  # noqa: D400
        return self.value // other

    def __divmod__(self, other):  # noqa: ANN001, ANN204
        """divmod"""  # noqa: D400, D403
        return divmod(self.value, other)

    def __mod__(self, other):  # noqa: ANN001, ANN204
        """%"""  # noqa: D400
        return self.value % other

    def __mul__(self, other):  # noqa: ANN001, ANN204
        """*"""  # noqa: D400
        return self.value * other

    def __pow__(self, other):  # noqa: ANN001, ANN204
        """**"""  # noqa: D400
        return self.value**other

    def __gt__(self, other):  # noqa: ANN001, ANN204
        """>"""  # noqa: D400
        return self.value > other

    def __ge__(self, other):  # noqa: ANN001, ANN204
        """>="""  # noqa: D400
        return self.value >= other

    def __le__(self, other):  # noqa: ANN001, ANN204
        """<="""  # noqa: D400
        return self.value <= other

    def __lt__(self, other):  # noqa: ANN001, ANN204
        """<"""  # noqa: D400
        return self.value < other

    def __eq__(self, other):  # noqa: ANN001, ANN204
        """=="""  # noqa: D400
        return self.value == other

    def __ne__(self, other):  # noqa: ANN001, ANN204
        """!="""  # noqa: D400
        return self.value != other

    def __radd__(self, other):  # noqa: ANN001, ANN204
        """+ (right)"""  # noqa: D400
        return other + self.value

    def __rtruediv__(self, other):  # noqa: ANN001, ANN204
        """/ (right)"""  # noqa: D400
        return other / self.value

    def __rdivmod__(self, other):  # noqa: ANN001, ANN204
        """divmod (right)"""  # noqa: D400, D403
        return divmod(other, self.value)

    def __rfloordiv__(self, other):  # noqa: ANN001, ANN204
        """// (right)"""  # noqa: D400
        return other // self.value

    def __rmod__(self, other):  # noqa: ANN001, ANN204
        """% (right)"""  # noqa: D400
        return other % self.value

    def __rmul__(self, other):  # noqa: ANN001, ANN204
        """* (right)"""  # noqa: D400
        return other * self.value

    def __rpow__(self, other):  # noqa: ANN001, ANN204
        """** (right)"""  # noqa: D400
        return other**self.value

    def __rsub__(self, other):  # noqa: ANN001, ANN204
        """- (right)"""  # noqa: D400
        return other - self.value

    def __repr_args__(self) -> _repr.ReprArgs:
        """Strip defaults from args shown in repr."""
        for key, val in super().__repr_args__():
            if key in self.model_fields and not nan_or_equal(val, self.model_fields[key].default):
                yield key, val

    def __hash__(self) -> int:
        """Hash function for the class."""
        return hash(repr(self))


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
    item_list: list,
    item_type: type | tuple[type, ...],
    default: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
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
