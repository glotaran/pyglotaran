"""The parameters class."""
from __future__ import annotations

from collections.abc import Generator
from textwrap import indent
from typing import TYPE_CHECKING
from typing import Any

import asteval
import numpy as np
import pandas as pd
from tabulate import tabulate

from glotaran.io import load_parameters
from glotaran.parameter.parameter import Parameter
from glotaran.utils.ipython import MarkdownStr
from glotaran.utils.sanitize import pretty_format_numerical

if TYPE_CHECKING:
    from glotaran.parameter.parameter_history import ParameterHistory


class ParameterNotFoundException(Exception):
    """Raised when a Parameter is not found."""

    def __init__(self, label: str):  # noqa: D107
        super().__init__(f"Cannot find parameter {label}")


class Parameters:
    """A container for :class:`Parameter`."""

    loader = load_parameters

    def __init__(self, parameters: dict[str, Parameter]):
        """Create :class:`Parameters`.

        Parameters
        ----------
        parameters : dict[str, Parameter]
            A parameter list containing parameters

        Returns
        -------
        'Parameters'
            The created :class:`Parameters`.
        """
        self._parameters: dict[str, Parameter] = parameters
        self._evaluator = asteval.Interpreter(symtable=asteval.make_symbol_table(parameters=self))
        self.source_path = "parameters.csv"
        self.update_parameter_expression()

    @classmethod
    def from_list(
        cls, parameter_list: list[float | int | str | list[Any] | dict[str, Any]]
    ) -> Parameters:
        """Create :class:`Parameters` from a list.

        Parameters
        ----------
        parameter_list : list[float | list[Any]]
            A parameter list containing parameters

        Returns
        -------
        Parameters
            The created :class:`Parameters`.

        .. # noqa: D414
        """
        defaults: dict[str, Any] | None = next(
            (item for item in parameter_list if isinstance(item, dict)), None
        )
        parameters = {}

        for i, item in enumerate(item for item in parameter_list if not isinstance(item, dict)):
            if not isinstance(item, list):
                item = [item]
            if not any(isinstance(v, str) for v in item):
                item += [f"{i+1}"]
            parameter = Parameter.from_list(item, default_options=defaults)
            parameters[parameter.label] = parameter
        return cls(parameters)

    @classmethod
    def from_dict(
        cls,
        parameter_dict: dict[str, dict[str, Any] | list[float | list[Any]]],
    ) -> Parameters:
        """Create a :class:`Parameters` from a dictionary.

        Parameters
        ----------
        parameter_dict: dict[str, dict[str, Any] | list[float | list[Any]]]
            A parameter dictionary containing parameters.

        Returns
        -------
        Parameters
            The created :class:`Parameters`

        .. # noqa: D414
        """
        parameters = {}
        for label, param_def, default in flatten_parameter_dict(parameter_dict):
            parameter = Parameter.from_list(param_def, default_options=default)
            label += f".{parameter.label}"
            parameter.label = label
            parameters[label] = parameter

        return cls(parameters)

    @classmethod
    def from_parameter_dict_list(cls, parameter_dict_list: list[dict[str, Any]]) -> Parameters:
        """Create :class:`Parameters` from a list of parameter dictionaries.

        Parameters
        ----------
        parameter_dict_list : list[dict[str, Any]]
            A list of parameter dictionaries.

        Returns
        -------
        Parameters
            The created :class:`Parameters`.

        .. # noqa: D414
        """
        parameters = {}
        for parameter_dict in parameter_dict_list:
            parameter = Parameter(**parameter_dict)
            parameters[parameter.label] = parameter
        return cls(parameters)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, source: str = "DataFrame") -> Parameters:
        """Create a :class:`Parameters` from a :class:`pandas.DataFrame`.

        Parameters
        ----------
        df : pd.DataFrame
            The source data frame.
        source : str
            Optional name of the source file, used for error messages.

        Raises
        ------
        ValueError
            Raised if the columns 'label' or 'value' doesn't exist. Also raised if the columns
            'minimum', 'maximum' or 'values' contain non numeric values or if the columns
            'non-negative' or 'vary' are no boolean.

        Returns
        -------
        Parameters
            The created parameter group.

        .. # noqa: D414
        """
        for column_name in ["label", "value"]:
            if column_name not in df:
                raise ValueError(f"Missing required column '{column_name}' in '{source}'.")

        for column_name in filter(lambda x: x in df.columns, ["minimum", "maximum", "value"]):
            if any(not np.isreal(v) for v in df[column_name]):
                raise ValueError(f"Column '{column_name}' in '{source}' has non numeric values.")

        for column_name in filter(lambda x: x in df.columns, ["non_negative", "vary"]):
            df[column_name] = [v != 0 if isinstance(v, int) else v for v in df[column_name]]
            if any(not isinstance(v, bool) for v in df[column_name]):
                raise ValueError(f"Column '{column_name}' in '{source}' has non boolean values.")

        # clean NaN if expressions
        if "expression" in df:
            expressions = df["expression"].to_list()
            df["expression"] = [expr if isinstance(expr, str) else None for expr in expressions]
        return cls.from_parameter_dict_list(df.to_dict(orient="records"))

    def to_dataframe(self) -> pd.DataFrame:
        """Create a pandas data frame from the group.

        Returns
        -------
        pd.DataFrame
            The created data frame.
        """
        return pd.DataFrame(self.to_parameter_dict_list())

    def to_parameter_dict_list(self) -> list[dict[str, Any]]:
        """Create list of parameter dictionaries from the group.

        Returns
        -------
        list[dict[str, Any]]
            A list of parameter dictionaries.
        """
        return [p.as_dict() for p in self.all()]

    def to_parameter_dict_or_list(self, serialize_parameters: bool = False) -> dict | list:
        """Convert to a dict or list of parameter definitions.

        Parameters
        ----------
        serialize_parameters : bool
            If true, the parameters will be serialized into list representation.

        Returns
        -------
        dict | list
            A dict or list of parameter definitions.
        """
        if all("." not in p.label for p in self.all()):
            return list(self.all())
        parameter_dict: dict[str, Any] = {}
        for parameter in self.all():
            path = parameter.label.split(".")
            nodes = path[:-2]
            node = parameter_dict
            for n in nodes:
                if n not in node:
                    node[n] = {}
                node = node[n]
            upper_node = path[-2]
            if upper_node not in node:
                node[upper_node] = []
            node[upper_node].append(
                parameter.as_list(label_short=True) if serialize_parameters else parameter
            )
        return parameter_dict

    def set_from_history(self, history: ParameterHistory, index: int):
        """Update the :class:`Parameters` with values from a parameter history.

        Parameters
        ----------
        history : ParameterHistory
            The parameter history.
        index : int
            The history index.
        """
        self.set_from_label_and_value_arrays(
            # Omit 0th element with `iteration` label
            history.parameter_labels[1:],
            history.get_parameters(index)[1:],
        )

    def copy(self) -> Parameters:
        """Create a copy of the :class:`Parameters`.

        Returns
        -------
        Parameters :
            A copy of the :class:`Parameters`.

        .. # noqa: D414
        """
        return Parameters(
            {label: parameter.copy() for label, parameter in self._parameters.items()}
        )

    def all(self) -> Generator[Parameter, None, None]:
        """Iterate over all parameters.

        Yields
        ------
        Parameter
            A parameter in the parameters.
        """
        yield from self._parameters.values()

    def has(self, label: str) -> bool:
        """Check if a parameter with the given label is in the group or in a subgroup.

        Parameters
        ----------
        label : str
            The label of the parameter, with its path in a :class:`ParameterGroup` prepended.

        Returns
        -------
        bool
            Whether a parameter with the given label exists in the group.
        """
        return label in self._parameters

    def get(self, label: str) -> Parameter:
        """Get a :class:`Parameter` by its label.

        Parameters
        ----------
        label : str
            The label of the parameter, with its path in a :class:`ParameterGroup` prepended.

        Returns
        -------
        Parameter
            The parameter.

        Raises
        ------
        ParameterNotFoundException
            Raised if no parameter with the given label exists.
        """
        try:
            return self._parameters[label]
        except KeyError as error:
            raise ParameterNotFoundException(label) from error

    def update_parameter_expression(self):
        """Update all parameters which have an expression.

        Raises
        ------
        ValueError
            Raised if an expression evaluates to a non-numeric value.
        """
        for parameter in self.all():
            if parameter.expression is not None:
                value = self._evaluator(parameter.transformed_expression)
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Expression '{parameter.expression}' of parameter '{parameter.label}' "
                        f"evaluates to non numeric value '{value}'."
                    )
                parameter.value = value

    def get_label_value_and_bounds_arrays(
        self, exclude_non_vary: bool = False
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        """Return a arrays of all parameter labels, values and bounds.

        Parameters
        ----------
        exclude_non_vary: bool
            If true, parameters with `vary=False` are excluded.

        Returns
        -------
        tuple[list[str], np.ndarray, np.ndarray, np.ndarray]
            A tuple containing a list of parameter labels and
            an array of the values, lower and upper bounds.
        """
        self.update_parameter_expression()

        labels = []
        values = []
        lower_bounds = []
        upper_bounds = []

        for parameter in self.all():
            if not exclude_non_vary or parameter.vary:
                labels.append(parameter.label)
                value, minimum, maximum = parameter.get_value_and_bounds_for_optimization()
                values.append(value)
                lower_bounds.append(minimum)
                upper_bounds.append(maximum)

        return labels, np.asarray(values), np.asarray(lower_bounds), np.asarray(upper_bounds)

    def set_from_label_and_value_arrays(self, labels: list[str], values: np.ndarray):
        """Update the parameter values from a list of labels and values.

        Parameters
        ----------
        labels : list[str]
            A list of parameter labels.
        values : np.ndarray
            An array of parameter values.

        Raises
        ------
        ValueError
            Raised if the size of the labels does not match the stize of values.
        """
        if len(labels) != len(values):
            raise ValueError(
                f"Length of labels({len(labels)}) not equal to length of values({len(values)})."
            )

        for label, value in zip(labels, values):
            self.get(label).set_value_from_optimization(value)

        self.update_parameter_expression()

    def markdown(self, float_format: str = ".3e") -> MarkdownStr:
        """Format the :class:`ParameterGroup` as markdown string.

        This is done by recursing the nested :class:`ParameterGroup` tree.

        Parameters
        ----------
        float_format: str
            Format string for floating point numbers, by default ".3e"

        Returns
        -------
        MarkdownStr :
            The markdown representation as string.
        """
        return param_dict_to_markdown(self.to_parameter_dict_or_list(), float_format=float_format)

    def _repr_markdown_(self) -> str:
        """Create a markdown representation.

        Special method used by ``ipython`` to render markdown.

        Returns
        -------
        str :
            The markdown representation as string.
        """
        return str(self.markdown())

    @property
    def labels(self) -> list[str]:
        """List of all labels.

        Returns
        -------
        list[str]
        """
        return sorted(p.label for p in self.all())

    def __str__(self) -> str:
        """Representation used by print and str."""
        return str(self.markdown())

    def __repr__(self) -> str:
        """Representation debug."""
        params = [f"{p.label!r}: {repr(p)}" for p in self.all()]
        return f"Parameters({{{', '.join(params)}}})"

    def __eq__(self, other: object) -> bool:
        """=="""  # noqa: D400
        if isinstance(other, Parameters):
            return self.labels == other.labels and all(
                self.get(label)._deep_equals(other.get(label)) for label in self.labels
            )
        raise NotImplementedError(
            "Parameters can only be compared with instances of Parameters, "
            f"not with {type(other).__qualname__!r}."
        )


def flatten_parameter_dict(
    parameter_dict: dict,
) -> Generator[tuple[str, list[Any], dict[str, Any] | None], None, None]:
    """Flatten a parameter dictionary.

    Parameters
    ----------
    parameter_dict: dict
        The parameter dictionary.

    Yields
    ------
    tuple[str, list[Any], dict | None
        The concatenated keys, the parameter definition and default options.
    """
    for key, value in parameter_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value, sub_dict in flatten_parameter_dict(value):
                yield f"{key}.{sub_key}", sub_value, sub_dict
        elif isinstance(value, list):
            sub_dict: dict[str, Any] | None = next(  # type:ignore[no-redef]
                (item for item in value if isinstance(item, dict)), None
            )
            for index, list_value in enumerate(
                (list_value for list_value in value if not isinstance(list_value, dict)), start=1
            ):
                if not isinstance(list_value, list):
                    list_value = [str(index), list_value]
                elif not any(isinstance(v, str) for v in list_value):
                    list_value += [str(index)]
                yield key, list_value, sub_dict


def param_dict_to_markdown(
    parameters: dict | list,
    float_format: str = ".3e",
    depth: int = 0,
    label: str | None = None,
) -> MarkdownStr:
    """Format the :class:`Parameters` as markdown string.

    This is done by recursing the nested :class:`Parameters` tree.

    Parameters
    ----------
    parameters: dict | list
        The parameter dict or list.
    float_format: str
        Format string for floating point numbers, by default ".3e"
    depth: int
        The depth of the parameter dict.
    label: str | None
        The label of the parameter dict.

    Returns
    -------
    MarkdownStr :
        The markdown representation as string.
    """
    node_indentation = "  " * depth
    return_string = ""
    table_header = [
        "_Label_",
        "_Value_",
        "_Standard Error_",
        "_t-value_",
        "_Minimum_",
        "_Maximum_",
        "_Vary_",
        "_Non-Negative_",
        "_Expression_",
    ]
    if label is not None:
        return_string += f"{node_indentation}* __{label}__:\n"
    if isinstance(parameters, list):
        parameter_rows = [
            [
                parameter.label_short,
                parameter.value,
                parameter.standard_error,
                repr(pretty_format_numerical(parameter.value / parameter.standard_error)),
                parameter.minimum,
                parameter.maximum,
                parameter.vary,
                parameter.non_negative,
                f"`{parameter.expression}`",
            ]
            for parameter in parameters
        ]
        parameter_table = indent(
            tabulate(
                parameter_rows,
                headers=table_header,
                tablefmt="github",
                missingval="None",
                floatfmt=float_format,
            ),
            f"  {node_indentation}",
        )
        return_string += f"\n{parameter_table}\n\n"
    else:
        for label, child in sorted(parameters.items()):
            return_string += str(
                param_dict_to_markdown(
                    child, float_format=float_format, depth=depth + 1, label=label
                )
            )
    return MarkdownStr(return_string.replace("'", " "))
