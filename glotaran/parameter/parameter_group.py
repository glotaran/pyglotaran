"""The parameter group class."""

from __future__ import annotations

from copy import copy
from textwrap import indent
from typing import TYPE_CHECKING
from typing import Generator

import asteval
import numpy as np
import pandas as pd
from tabulate import tabulate

from glotaran.deprecation import deprecate
from glotaran.io import save_parameters
from glotaran.parameter.parameter import Parameter
from glotaran.utils.ipython import MarkdownStr

if TYPE_CHECKING:
    from glotaran.parameter.parameter_history import ParameterHistory


class ParameterNotFoundException(Exception):
    """Raised when a Parameter is not found in the Group."""

    def __init__(self, path, label):  # noqa: D107
        super().__init__(f"Cannot find parameter {'.'.join(path+[label])!r}")


class ParameterGroup(dict):
    """Represents are group of parameters.

    Can contain other groups, creating a tree-like hierarchy.
    """

    def __init__(self, label: str = None, root_group: ParameterGroup = None):
        """Initialize a :class:`ParameterGroup` instance with ``label``.

        Parameters
        ----------
        label : str
            The label of the group.
        root_group : ParameterGroup
            The root group

        Raises
        ------
        ValueError
            Raised if the an invalid label is given.
        """
        if label is not None and not Parameter.valid_label(label):
            raise ValueError(f"'{label}' is not a valid group label.")
        self._label = label
        self._parameters: dict[str, Parameter] = {}
        self._root_group = root_group
        self._evaluator = (
            asteval.Interpreter(symtable=asteval.make_symbol_table(group=self))
            if root_group is None
            else None
        )
        super().__init__()

    @classmethod
    def from_dict(
        cls,
        parameter_dict: dict[str, dict | list],
        label: str = None,
        root_group: ParameterGroup = None,
    ) -> ParameterGroup:
        """Create a :class:`ParameterGroup` from a dictionary.

        Parameters
        ----------
        parameter_dict : dict[str, dict | list]
            A parameter dictionary containing parameters.
        label : str
            The label of the group.
        root_group : ParameterGroup
            The root group

        Returns
        -------
        ParameterGroup
            The created :class:`ParameterGroup`
        """
        root = cls(label=label, root_group=root_group)
        for label, item in parameter_dict.items():
            label = str(label)
            if isinstance(item, dict):
                root.add_group(cls.from_dict(item, label=label, root_group=root))
            if isinstance(item, list):
                root.add_group(cls.from_list(item, label=label, root_group=root))
        if root_group is None:
            root.update_parameter_expression()
        return root

    @classmethod
    def from_list(
        cls,
        parameter_list: list[float | list],
        label: str = None,
        root_group: ParameterGroup = None,
    ) -> ParameterGroup:
        """Create a :class:`ParameterGroup` from a list.

        Parameters
        ----------
        parameter_list : list[float | list]
            A parameter list containing parameters
        label : str
            The label of the group.
        root_group : ParameterGroup
            The root group

        Returns
        -------
        ParameterGroup
            The created :class:`ParameterGroup`
        """
        root = cls(label=label, root_group=root_group)

        # get defaults
        defaults = None
        for item in parameter_list:
            if isinstance(item, dict):
                defaults = item
                break

        for i, item in enumerate(parameter_list):
            if isinstance(item, (str, int, float)):
                try:
                    item = float(item)
                except Exception:
                    pass
            if isinstance(item, (float, int, list)):
                root.add_parameter(
                    Parameter.from_list_or_value(item, label=str(i + 1), default_options=defaults)
                )
        if root_group is None:
            root.update_parameter_expression()
        return root

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, source: str = "DataFrame") -> ParameterGroup:
        """Create a :class:`ParameterGroup` from a :class:`pandas.DataFrame`.

        Parameters
        ----------
        df : pd.DataFrame
            The source data frame.
        source : str
            Optional name of the source file, used for error messages.

        Returns
        -------
        ParameterGroup
            The created parameter group.

        Raises
        ------
        ValueError
            Raised if the columns 'label' or 'value' doesn't exist. Also raised if the columns
            'minimum', 'maximum' or 'values' contain non numeric values or if the columns
            'non-negative' or 'vary' are no boolean.
        """
        for column_name in ["label", "value"]:
            if column_name not in df:
                raise ValueError(f"Missing column '{column_name}' in '{source}'")

        for column_name in ["minimum", "maximum", "value"]:
            if column_name in df and any(not np.isreal(v) for v in df[column_name]):
                raise ValueError(f"Column '{column_name}' in '{source}' has non numeric values")

        for column_name in ["non-negative", "vary"]:
            if column_name in df and any(not isinstance(v, bool) for v in df[column_name]):
                raise ValueError(f"Column '{column_name}' in '{source}' has non boolean values")

        root = cls()

        for i, full_label in enumerate(df["label"]):
            path = full_label.split(".")
            group = root
            while len(path) > 1:
                group_label = path.pop(0)
                if group_label not in group:
                    group.add_group(ParameterGroup(label=group_label, root_group=group))
                group = group[group_label]
            label = path.pop()
            value = df["value"][i]
            minimum = df["minimum"][i] if "minimum" in df else -np.inf
            maximum = df["maximum"][i] if "maximum" in df else np.inf
            non_negative = df["non-negative"][i] if "non-negative" in df else False
            vary = df["vary"][i] if "vary" in df else True
            expression = (
                df["expression"][i]
                if "expression" in df and isinstance(df["expression"][i], str)
                else None
            )

            parameter = Parameter(
                label=label,
                full_label=full_label,
                value=value,
                expression=expression,
                maximum=maximum,
                minimum=minimum,
                non_negative=non_negative,
                vary=vary,
            )
            group.add_parameter(parameter)
        root.update_parameter_expression()
        return root

    @property
    def label(self) -> str | None:
        """Label of the group.

        Returns
        -------
        str
            The label of the group.
        """
        return self._label

    @property
    def root_group(self) -> ParameterGroup | None:
        """Root of the group.

        Returns
        -------
        ParameterGroup
            The root group.
        """
        return self._root_group

    def to_dataframe(self) -> pd.DataFrame:
        """Create a pandas data frame from the group.

        Returns
        -------
        pd.DataFrame
            The created data frame.
        """
        parameter_dict: dict[str, list[str | float | bool | None]] = {
            "label": [],
            "value": [],
            "minimum": [],
            "maximum": [],
            "vary": [],
            "non-negative": [],
            "expression": [],
        }
        for label, parameter in self.all():
            parameter_dict["label"].append(label)
            parameter_dict["value"].append(parameter.value)
            parameter_dict["minimum"].append(parameter.minimum)
            parameter_dict["maximum"].append(parameter.maximum)
            parameter_dict["vary"].append(parameter.vary)
            parameter_dict["non-negative"].append(parameter.non_negative)
            parameter_dict["expression"].append(parameter.expression)
        return pd.DataFrame(parameter_dict)

    @deprecate(
        deprecated_qual_name_usage=(
            "glotaran.parameter.ParameterGroup.to_csv(file_name=<parameters.csv>)"
        ),
        new_qual_name_usage=(
            "glotaran.io.save_parameters(parameters, "
            'file_name=<parameters.csv>, format_name="csv")'
        ),
        to_be_removed_in_version="0.7.0",
        importable_indices=(2, 1),
    )
    def to_csv(self, filename: str, delimiter: str = ",") -> None:
        """Save a :class:`ParameterGroup` to a CSV file.

        Warning
        -------
        Deprecated use
        ``glotaran.io.save_parameters(parameters, file_name=<parameters.csv>, format_name="csv")``
        instead.

        Parameters
        ----------
        filename : str
            File to write the parameter specs to.
        delimiter : str
            Character to separate columns., by default ","
        """
        save_parameters(self, file_name=filename, allow_overwrite=True, sep=delimiter)

    def add_parameter(self, parameter: Parameter | list[Parameter]):
        """Add a :class:`Parameter` to the group.

        Parameters
        ----------
        parameter : Parameter | list[Parameter]
            The parameter to add.

        Raises
        ------
        TypeError
            If ``parameter`` or any item of it is not an instance of :class:`Parameter`.
        """
        if not isinstance(parameter, list):
            parameter = [parameter]
        if any(not isinstance(p, Parameter) for p in parameter):
            raise TypeError("Parameter must be  instance of glotaran.parameter.Parameter")
        for p in parameter:
            p.index = len(self._parameters) + 1
            if p.label is None:
                p.label = f"{p.index}"
            p.full_label = f"{self.label}.{p.label}" if self.label else p.label
            self._parameters[p.label] = p

    def add_group(self, group: ParameterGroup):
        """Add a :class:`ParameterGroup` to the group.

        Parameters
        ----------
        group : ParameterGroup
            The group to add.

        Raises
        ------
        TypeError
            Raised if the group is not an instance of :class:`ParameterGroup`.
        """
        if not isinstance(group, ParameterGroup):
            raise TypeError("Group must be glotaran.parameter.ParameterGroup")
        self[group.label] = group

    def get_nr_roots(self) -> int:
        """Return the number of roots of the group.

        Returns
        -------
        int
            The number of roots.
        """
        n = 0
        root = self.root_group
        while root is not None:
            n += 1
            root = root.root_group
        return n

    def groups(self) -> Generator[ParameterGroup, None, None]:
        """Return a generator over all groups and their subgroups.

        Yields
        ------
        ParameterGroup
            A subgroup of :class:`ParameterGroup`.
        """
        for group in self:
            yield from group.groups()

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
        try:
            self.get(label)
            return True
        except Exception:
            return False

    def get(self, label: str) -> Parameter:  # type:ignore[override]
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
        # sometimes the spec parser delivers the labels as int
        label = str(label)

        path = label.split(".")
        label = path.pop()

        # TODO: audit this code
        group = self
        for element in path:
            try:
                group = group[element]
            except KeyError:
                raise ParameterNotFoundException(path, label)
        try:
            return group._parameters[label]
        except KeyError:
            raise ParameterNotFoundException(path, label)

    def copy(self) -> ParameterGroup:
        """Create a copy of the :class:`ParameterGroup`.

        Returns
        -------
        ParameterGroup :
            A copy of the :class:`ParameterGroup`.

        """
        root = ParameterGroup(label=self.label, root_group=self.root_group)

        for label, parameter in self._parameters.items():
            root._parameters[label] = copy(parameter)

        for label, group in self.items():
            root[label] = group.copy()

        return root

    def all(
        self, root: str | None = None, separator: str = "."
    ) -> Generator[tuple[str, Parameter], None, None]:
        """Iterate over all parameter in the group and it's subgroups together with their labels.

        Parameters
        ----------
        root : str
            The label of the root group
        separator : str
            The separator for the parameter labels.

        Yields
        ------
        tuple[str, Parameter]
            A tuple containing the full label of the parameter and the parameter itself.
        """
        root = f"{root}{self.label}{separator}" if root is not None else ""
        for label, p in self._parameters.items():
            yield (f"{root}{label}", p)
        for _, l in self.items():
            yield from l.all(root=root, separator=separator)

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

        for label, parameter in self.all():
            if not exclude_non_vary or parameter.vary:
                labels.append(label)
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

    def set_from_history(self, history: ParameterHistory, index: int):
        """Update the :class:`ParameterGroup` with values from a parameter history.

        Parameters
        ----------
        history : ParameterHistory
            The parameter history.
        index : int
            The history index.
        """
        self.set_from_label_and_value_arrays(
            history.parameter_labels, history.get_parameters(index)
        )

    def update_parameter_expression(self):
        """Update all parameters which have an expression.

        Raises
        ------
        ValueError
            Raised if an expression evaluates to a non-numeric value.
        """
        for label, parameter in self.all():
            if parameter.expression is not None:
                value = self._evaluator(parameter.transformed_expression)
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Expression '{parameter.expression}' of parameter '{label}' evaluates to "
                        f"non numeric value '{value}'."
                    )
                parameter.value = value

    def markdown(self) -> MarkdownStr:
        """Format the :class:`ParameterGroup` as markdown string.

        This is done by recursing the nested :class:`ParameterGroup` tree.

        Returns
        -------
        MarkdownStr :
            The markdown representation as string.
        """
        node_indentation = "  " * self.get_nr_roots()
        return_string = ""
        table_header = [
            "_Label_",
            "_Value_",
            "_StdErr_",
            "_Min_",
            "_Max_",
            "_Vary_",
            "_Non-Negative_",
            "_Expr_",
        ]
        if self.label is not None:
            return_string += f"{node_indentation}* __{self.label}__:\n"
        if len(self._parameters):
            parameter_rows = [
                [
                    parameter.label,
                    parameter.value,
                    parameter.standard_error,
                    parameter.minimum,
                    parameter.maximum,
                    parameter.vary,
                    parameter.non_negative,
                    parameter.expression,
                ]
                for _, parameter in self._parameters.items()
            ]
            parameter_table = indent(
                tabulate(
                    parameter_rows, headers=table_header, tablefmt="github", missingval="None"
                ),
                f"  {node_indentation}",
            )
            return_string += f"\n{parameter_table}\n\n"
        for _, child_group in sorted(self.items()):
            return_string += f"{child_group.__str__()}"
        return MarkdownStr(return_string)

    def _repr_markdown_(self) -> str:
        """Create a markdown respresentation.

        Special method used by ``ipython`` to render markdown.

        Returns
        -------
        str :
            The markdown representation as string.
        """
        return str(self.markdown())

    def __repr__(self) -> str:
        """Representation used by repl and tracebacks.

        Returns
        -------
        str :
            A string representation of the :class:`ParameterGroup`.
        """
        parameter_short_notations = [
            [str(parameter.label), parameter.value] for parameter in self._parameters.values()
        ]
        if self.label is None:
            if len(self._parameters) == 0:
                return f"{type(self).__name__}.from_dict({super().__repr__()})"
            else:
                return f"{type(self).__name__}.from_list({parameter_short_notations})"
        if len(self._parameters):
            return parameter_short_notations.__repr__()
        else:
            return super().__repr__()

    def __str__(self) -> str:
        """Representation used by print and str."""
        return str(self.markdown())
