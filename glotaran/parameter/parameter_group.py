"""The parameter group class"""

from __future__ import annotations

import csv
import pathlib
from copy import copy
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple
from typing import Union

import asteval
import numpy as np
import pandas as pd
import yaml

from .parameter import Parameter


class ParameterNotFoundException(Exception):
    """Raised when a Parameter is not found in the Group."""

    def __init__(self, path, label):
        super().__init__(f"Cannot find parameter {'.'.join(path)}.{label}")


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
        parameter: Dict[str, Union[Dict, List]],
        label: str = None,
        root_group: ParameterGroup = None,
    ) -> ParameterGroup:
        """Creates a :class:`ParameterGroup` from a dictionary.

        Parameters
        ----------
        parameter :
            The parameter dictionary.
        label :
            The label of root group.
        root_group:
            The root group
        """
        root = cls(label=label, root_group=root_group)
        for label, item in parameter.items():
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
        parameter: List[Union[float, List]],
        label: str = None,
        root_group: ParameterGroup = None,
    ) -> ParameterGroup:
        """Creates a :class:`ParameterGroup` from a list.

        Parameters
        ----------
        parameter :
            The parameter list.
        label :
            The label of the root group.
        root_group:
            The root group
        """
        root = cls(label=label, root_group=root_group)

        # get defaults
        defaults = None
        for item in parameter:
            if isinstance(item, dict):
                defaults = item
                break

        for i, item in enumerate(parameter):
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
    def known_formats(cls) -> Dict[str, Callable]:
        return {
            "csv": cls.from_csv,
            "yml": cls.from_yaml_file,
            "yaml": cls.from_yaml_file,
        }

    @classmethod
    def from_file(cls, filepath: str, fmt: str = None):
        if fmt is None:
            path = pathlib.Path(filepath)
            fmt = path.suffix[1:] if path.suffix != "" else "yml"
        if fmt not in cls.known_formats():
            raise Exception(
                f"Unknown parameter format '{format}'. "
                f"Valid Formats are {cls.known_formats().keys()}."
            )
        return cls.known_formats()[fmt](filepath)

    @classmethod
    def from_yaml_file(cls, filepath: str) -> "ParameterGroup":
        """Creates a :class:`ParameterGroup` from a YAML file.

        Parameters
        ----------
        filepath :
            The path to the YAML file.
        """

        with open(filepath) as f:
            cls = cls.from_yaml(f)
        return cls

    @classmethod
    def from_yaml(cls, yaml_string: str):
        """Creates a :class:`ParameterGroup` from a YAML string.

        Parameters
        ----------
        yaml_string :
            The YAML string with the parameters.
        """
        items = yaml.safe_load(yaml_string)
        if isinstance(items, list):
            return cls.from_list(items)
        else:
            return cls.from_dict(items)

    @classmethod
    def from_csv(cls, filepath: str, delimiter: str = "\t"):
        """Creates a :class:`ParameterGroup` from a CSV file.

        Parameters
        ----------
        filepath :
            The path to the CSV file.
        delimiter : str
            The delimiter of the CSV file.
        """

        root = cls()
        df = pd.read_csv(filepath, sep=delimiter)

        for i, label in enumerate(df["label"]):
            label = label.split(".")
            if len(label) == 1:
                p = Parameter(label=label.pop())
                p.value = df["value"][i]
                p.stderr = df["stderr"][i]
                p.minimum = df["min"][i]
                p.maximum = df["max"][i]
                p.vary = df["vary"][i]
                p.non_negative = df["non-negative"][i]
                root.add_parameter(p)
                continue

            top = root
            while len(label) != 0:
                group = label.pop(0)
                if group in top:
                    if len(label) == 1:
                        p = Parameter(label=label.pop())
                        p.value = df["value"][i]
                        p.stderr = df["stderr"][i]
                        p.minimum = df["min"][i]
                        p.maximum = df["max"][i]
                        p.vary = df["vary"][i]
                        p.non_negative = df["non-negative"][i]
                        top[group].add_parameter(p)
                    else:
                        top = top[group]
                else:
                    group = ParameterGroup(group)
                    top.add_group(group)
                    if len(label) == 1:
                        p = Parameter(label=label.pop())
                        p.value = df["value"][i]
                        p.stderr = df["stderr"][i]
                        p.minimum = df["min"][i]
                        p.maximum = df["max"][i]
                        p.vary = df["vary"][i]
                        p.non_negative = df["non-negative"][i]
                        group.add_parameter(p)
                    else:
                        top = group
        return root

    @property
    def label(self) -> str:
        """Label of the group."""
        return self._label

    @property
    def root_group(self) -> ParameterGroup:
        """Root of the group."""
        return self._root_group

    def to_csv(self, filename: str, delimiter: str = "\t"):
        """Writes a :class:`ParameterGroup` to a CSV file.

        Parameters
        ----------
        filepath :
            The path to the CSV file.
        delimiter : str
            The delimiter of the CSV file.
        """

        with open(filename, mode="w") as parameter_file:
            parameter_writer = csv.writer(parameter_file, delimiter=delimiter)
            parameter_writer.writerow(
                ["label", "value", "min", "max", "vary", "non-negative", "stderr"]
            )

            for (label, p) in self.all():
                parameter_writer.writerow(
                    [label, p.value, p.minimum, p.maximum, p.vary, p.non_negative, p.stderr]
                )

    def add_parameter(self, parameter: Parameter):
        """Adds a :class:`Parameter` to the group.

        Parameters
        ----------
        parameter :
            The parameter to add.
        """
        if not isinstance(parameter, list):
            parameter = [parameter]
        if any(not isinstance(p, Parameter) for p in parameter):
            raise TypeError("Parameter must be  instance of glotaran.model.Parameter")
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
            raise TypeError("Group must be glotaran.model.ParameterGroup")
        self[group.label] = group

    def get_nr_roots(self) -> int:
        """Returns the number of roots of the group."""
        n = 0
        root = self.root_group
        while root is not None:
            n += 1
            root = root.root_group
        return n

    def groups(self) -> Generator["ParameterGroup", None, None]:
        """Returns a generator over all groups and their subgroups."""
        for group in self:
            yield from group.groups()

    def has(self, label: str) -> bool:
        """Checks if a parameter with the given label is in the group or in a subgroup.

        Parameters
        ----------
        label :
            The label of the parameter.
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
            The label of the parameter.
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
    ) -> Generator[Tuple[str, Parameter], None, None]:
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
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
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

    def set_from_label_and_value_arrays(self, labels: List[str], values: np.ndarray):
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

    def markdown(self) -> str:
        """Formats the :class:`ParameterGroup` as markdown string."""
        t = "".join("  " for _ in range(self.get_nr_roots()))
        s = ""
        if self.label != "p":
            s += f"{t}* __{self.label}__:\n"
        for _, p in self._parameters.items():
            s += f"{t}  * {p}\n"
        for _, g in self.items():
            s += f"{g.__str__()}"
        return s

    def __repr__(self):
        return self.markdown()

    def __str__(self):
        return self.__repr__()
