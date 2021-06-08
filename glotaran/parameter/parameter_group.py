"""The parameter group class"""

from __future__ import annotations

from copy import copy
from textwrap import indent
from typing import Generator

import asteval
import numpy as np
import pandas as pd
from tabulate import tabulate

from glotaran.parameter.parameter import Parameter
from glotaran.utils.ipython import MarkdownStr


class ParameterNotFoundException(Exception):
    """Raised when a Parameter is not found in the Group."""

    def __init__(self, path, label):
        super().__init__(f"Cannot find parameter {'.'.join(path+[label])!r}")


class ParameterGroup(dict):
    def __init__(self, label: str = None, root_group: ParameterGroup = None):
        """Represents are group of parameters. Can contain other groups, creating a
        tree-like hierarchy.

        Parameters
        ----------
        label :
            The label of the group.
        """
        if label is not None and not Parameter.valid_label(label):
            raise ValueError(f"'{label}' is not a valid group label.")
        self._label = label
        self._parameters = {}
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
        """Creates a :class:`ParameterGroup` from a dictionary.

        Parameters
        ----------
        parameter_dict :
            A parameter dictionary containing parameters.
        label :
            The label of root group.
        root_group:
            The root group
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
        """Creates a :class:`ParameterGroup` from a list.

        Parameters
        ----------
        parameter_list :
            A parameter list containing parameters
        label :
            The label of the root group.
        root_group:
            The root group
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
        """Creates a :class:`ParameterGroup` from a :class:`pandas.DataFrame`"""

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
    def label(self) -> str:
        """Label of the group."""
        return self._label

    @property
    def root_group(self) -> ParameterGroup:
        """Root of the group."""
        return self._root_group

    def to_dataframe(self) -> pd.DataFrame:
        parameter_dict = {
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

    def to_csv(self, filename: str, delimiter: str = ","):
        """Writes a :class:`ParameterGroup` to a CSV file.

        Parameters
        ----------
        filepath :
            The path to the CSV file.
        delimiter : str
            The delimiter of the CSV file.
        """
        self.to_dataframe().to_csv(filename, sep=delimiter, na_rep="None", index=False)

    def add_parameter(self, parameter: Parameter | list[Parameter]):
        """Adds a :class:`Parameter` to the group.

        Parameters
        ----------
        parameter :
            The parameter to add.
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
        """Adds a :class:`ParameterGroup` to the group.

        Parameters
        ----------
        group :
            The group to add.
        """
        if not isinstance(group, ParameterGroup):
            raise TypeError("Group must be glotaran.parameter.ParameterGroup")
        self[group.label] = group

    def get_nr_roots(self) -> int:
        """Returns the number of roots of the group."""
        n = 0
        root = self.root_group
        while root is not None:
            n += 1
            root = root.root_group
        return n

    def groups(self) -> Generator[ParameterGroup, None, None]:
        """Returns a generator over all groups and their subgroups."""
        for group in self:
            yield from group.groups()

    def has(self, label: str) -> bool:
        """Checks if a parameter with the given label is in the group or in a subgroup.

        Parameters
        ----------
        label :
            The label of the parameter, with its path in a parameter group prepended.
        """

        try:
            self.get(label)
            return True
        except Exception:
            return False

    def get(self, label: str) -> Parameter:
        """Gets a :class:`Parameter` by its label.

        Parameters
        ----------
        label :
            The label of the parameter, with its path in a parameter group prepended.
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
        root = ParameterGroup(label=self.label, root_group=self.root_group)

        for label, parameter in self._parameters.items():
            root._parameters[label] = copy(parameter)

        for label, group in self.items():
            root[label] = group.copy()

        return root

    def all(
        self, root: str = None, separator: str = "."
    ) -> Generator[tuple[str, Parameter], None, None]:
        """Returns a generator over all parameter in the group and it's subgroups together with
        their labels.

        Parameters
        ----------
        root :
            The label of the root group
        separator:
            The separator for the parameter labels.
        """

        root = f"{root}{self.label}{separator}" if root is not None else ""
        for label, p in self._parameters.items():
            yield (f"{root}{label}", p)
        for _, l in self.items():
            yield from l.all(root=root, separator=separator)

    def get_label_value_and_bounds_arrays(
        self, exclude_non_vary: bool = False
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        """Returns a arrays of all parameter labels, values and bounds.

        Parameters
        ----------

        exclude_non_vary: bool = False
            If true, parameters with `vary=False` are excluded.
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
        """Updates the parameter values from a list of labels and values."""

        if len(labels) != len(values):
            raise ValueError(
                f"Length of labels({len(labels)}) not equal to length of values({len(values)})."
            )

        for label, value in zip(labels, values):
            self.get(label).set_value_from_optimization(value)

        self.update_parameter_expression()

    def update_parameter_expression(self):
        """Updates all parameters which have an expression."""
        for label, parameter in self.all():
            if parameter.expression is not None:
                value = self._evaluator(parameter.transformed_expression)
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Expression '{parameter.expression}' of parameter '{label}' evaluates to"
                        f"non numeric value '{value}'."
                    )
                parameter.value = value

    def markdown(self) -> MarkdownStr:
        """Formats the :class:`ParameterGroup` as markdown string.

        This is done by recursing the nested :class:`ParameterGroup` tree.
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
        """Special method used by ``ipython`` to render markdown."""
        return str(self.markdown())

    def __repr__(self):
        """Representation used by repl and tracebacks."""
        if self.label is None:
            return f"{type(self).__name__}.from_dict({super().__repr__()})"
        elif len(self._parameters):
            parameter_short_notations = [
                f"[{parameter.label!r}, {parameter.value}]"
                for _, parameter in self._parameters.items()
            ]
            return f"[{', '.join(parameter_short_notations)}]"
        else:
            return super().__repr__()

    def __str__(self):
        """Representation used by print and str."""
        return str(self.markdown())
