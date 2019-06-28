"""The parameter group class"""

from lmfit import Parameters
from math import log
import copy
import csv
import numpy as np
import pandas as pd
import pathlib
import typing
import yaml

from .parameter import Parameter


class ParameterNotFoundException(Exception):
    """Raised when a Parameter is not found in the Group."""
    def __init__(self, path, label):
        super(ParameterNotFoundException, self).__init__(
            f"Cannot find parameter {'.'.join(path)}.{label}")


class ParameterGroup(dict):

    def __init__(self, label: str = None):
        """Represents are group of parameters. Can contain other groups, creating a
        tree-like hirachy.

        Parameters
        ----------
        label :
            The label of the group.
        """

        self._label = label
        self._parameters = {}
        self._root = None
        super(ParameterGroup, self).__init__()

    @classmethod
    def from_parameter_dict(cls, parameter: Parameters):
        """Creates a :class:`ParameterGroup` from an lmfit.Parameters dictionary

        Parameters
        ----------
        parameter :
            A lmfit.Parameters dictionary
        """

        root = cls(None)
        for lbl, param in parameter.items():
            lbl = lbl.split("_")
            if len(lbl) == 2:
                # it is a root param
                param = Parameter.from_parameter(lbl.pop(), param)
                root.add_parameter(param)
                continue

            # remove root
            lbl.pop(0)

            top = root
            while len(lbl) != 0:
                group = lbl.pop(0)
                if group in top:
                    if len(lbl) == 1:
                        param = Parameter.from_parameter(lbl.pop(), param)
                        top[group].add_parameter(param)
                    else:
                        top = top[group]
                else:
                    group = ParameterGroup(group)
                    top.add_group(group)
                    if len(lbl) == 1:
                        param = Parameter.from_parameter(lbl.pop(), param)
                        group.add_parameter(param)
                    else:
                        top = group
        return root

    @classmethod
    def from_dict(cls,
                  parameter: typing.Dict[str, typing.Union[typing.Dict, typing.List]],
                  label="p") -> "ParameterGroup":
        """Creates a :class:`ParameterGroup` from a dictionary.

        Parameters
        ----------
        parameter :
            The parameter dictionary.
        label :
            The label of root group.
        """
        root = cls(label)
        for label, item in parameter.items():
            label = str(label)
            if isinstance(item, dict):
                root.add_group(cls.from_dict(item, label=label))
            if isinstance(item, list):
                root.add_group(cls.from_list(item, label=label))
        return root

    @classmethod
    def from_list(cls,
                  parameter: typing.List[typing.Union[float, typing.List]],
                  label="p") -> "ParameterGroup":
        """Creates a :class:`ParameterGroup` from a list.

        Parameters
        ----------
        parameter :
            The parameter list.
        label :
            The label of the root group.
        """
        root = cls(label)

        # get defaults
        defaults = None
        for item in parameter:
            if isinstance(item, dict):
                defaults = item
                break

        for item in parameter:
            if isinstance(item, str):
                try:
                    item = float(item)
                except Exception:
                    pass
            if isinstance(item, (float, int, list)):
                root.add_parameter(Parameter.from_list_or_value(item, default_options=defaults))
        return root

    @classmethod
    def known_formats(cls) -> typing.Dict[str, typing.Callable]:
        return {
            'csv': cls.from_csv,
            'yml': cls.from_yaml_file,
            'yaml': cls.from_yaml_file,
        }

    @classmethod
    def from_file(cls, filepath: str, fmt: str = None):
        if fmt is None:
            path = pathlib.Path(filepath)
            fmt = path.suffix[1:] if path.suffix != '' else 'yml'
        if fmt not in cls.known_formats():
            raise Exception(f"Unknown parameter format '{format}'. "
                            f"Valid Formats are {cls.known_formats().keys()}.")
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
        items = yaml.load(yaml_string, Loader=yaml.FullLoader)
        if isinstance(items, list):
            cls = cls.from_list(items)
        else:
            cls = cls.from_dict(items)
        return cls

    @classmethod
    def from_csv(cls, filepath: str, delimiter: str = '\t'):
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

        for i, label in enumerate(df['label']):
            label = label.split('.')
            if len(label) == 1:
                p = Parameter(label=label.pop())
                p.value = df['value'][i]
                p.stderr = df['stderr'][i]
                p.min = df['min'][i]
                p.max = df['max'][i]
                p.vary = df['vary'][i]
                p.non_neg = df['non-negative'][i]
                root.add_parameter(p)
                continue

            top = root
            while len(label) != 0:
                group = label.pop(0)
                if group in top:
                    if len(label) == 1:
                        p = Parameter(label=label.pop())
                        p.value = df['value'][i]
                        p.stderr = df['stderr'][i]
                        p.min = df['min'][i]
                        p.max = df['max'][i]
                        p.vary = df['vary'][i]
                        p.non_neg = df['non-negative'][i]
                        top[group].add_parameter(p)
                    else:
                        top = top[group]
                else:
                    group = ParameterGroup(group)
                    top.add_group(group)
                    if len(label) == 1:
                        p = Parameter(label=label.pop())
                        p.value = df['value'][i]
                        p.stderr = df['stderr'][i]
                        p.min = df['min'][i]
                        p.max = df['max'][i]
                        p.vary = df['vary'][i]
                        p.non_neg = df['non-negative'][i]
                        group.add_parameter(p)
                    else:
                        top = group
        return root

    def to_csv(self, filename: str, delimiter: str = '\t'):
        """Writes a :class:`ParameterGroup` to a CSV file.

        Parameters
        ----------
        filepath :
            The path to the CSV file.
        delimiter : str
            The delimiter of the CSV file.
        """

        with open(filename, mode='w') as parameter_file:
            parameter_writer = csv.writer(parameter_file, delimiter=delimiter)
            parameter_writer.writerow(
                ['label', 'value', 'min', 'max', 'vary', 'non-negative', 'stderr']
            )

            for (label, p) in self.all():
                parameter_writer.writerow(
                    [label, p.value, p.min, p.max, p.vary, p.non_neg, p.stderr]
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
            raise TypeError("Parameter must be  instance of"
                            " glotaran.model.Parameter")
        for p in parameter:
            p.index = len(self._parameters) + 1
            if p.label is None:
                p.label = f"{p.index}"
            p.full_label = f'{self.label}.{p.label}' if self.label else p.label
            self._parameters[p.label] = p

    def add_group(self, group: 'ParameterGroup'):
        """Adds a :class:`ParameterGroup` to the group.

        Parameters
        ----------
        group :
            The group to add.
        """
        if not isinstance(group, ParameterGroup):
            raise TypeError("Group must be glotaran.model.ParameterGroup")
        group.set_root(self)
        self[group.label] = group

    def set_root(self, root: 'ParameterGroup'):
        """Sets the root of the group.

        Parameters
        ----------
        root :
            The new root of the group.
        """
        self._root = root

    def get_nr_roots(self) -> int:
        """Returns the number of roots of the group."""
        n = 0
        root = self._root
        while root is not None:
            n += 1
            root = root._root
        return n

    @property
    def label(self) -> str:
        """Label of the group """
        return self._label

    def groups(self) -> typing.Generator['ParameterGroup', None, None]:
        """Returns a generator over all groups and their subgroups."""
        for group in self:
            for l in group.groups():
                yield l

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
        """Gets a :class:`Parameter` by it label.

        Parameters
        ----------
        label :
            The label of the parameter.
        """

        # sometimes the spec parser delivers the labels as int
        label = str(label)

        path = label.split(".")
        label = path.pop()

        group = self
        for l in path:
            try:
                group = group[l]
            except KeyError:
                raise ParameterNotFoundException(path, label)
        try:
            return group._parameters[label]
        except KeyError:
            raise ParameterNotFoundException(path, label)

    def all(self, root: str = None, seperator: str = ".") \
            -> typing.Generator[typing.Tuple[str, Parameter], None, None]:
        """Returns a generator over all parameter in the group and it's subgroups together with
        their labels.

        Parameters
        ----------
        root :
            The label of the root group
        seperator:
            The seperator for the parameter labels.
        """

        root = f"{root}{self.label}{seperator}" if root is not None else ""
        for label, p in self._parameters.items():
            yield (f"{root}{label}", p)
        for _, l in self.items():
            for (lbl, p) in l.all(root=root, seperator=seperator):
                yield (lbl, p)

    def as_parameter_dict(self) -> Parameters:
        """
        Creates a lmfit.Parameters dictionary from the group.

        Notes
        -----

        Only for internal use.
        """

        params = Parameters()
        for label, p in self.all(seperator="_"):
            p.name = "_" + label
            if p.non_neg:
                p = copy.deepcopy(p)
                if p.value == 1:
                    p.value += 1e-10
                if p.min == 1:
                    p.min += 1e-10
                if p.max == 1:
                    p.max += 1e-10
                else:
                    try:
                        p.value = log(p.value)
                        p.min = log(p.min) if np.isfinite(p.min) else p.min
                        p.max = log(p.max) if np.isfinite(p.max) else p.max
                    except Exception:
                        raise Exception("Could not take log of parameter"
                                        f" '{label}' with value '{p.value}'")
            params.add(p)
        return params

    def markdown(self) -> str:
        """Formats the :class:`ParameterGroup` as markdown string."""
        t = "".join(["  " for _ in range(self.get_nr_roots())])
        s = ""
        if self.label != "p":
            s += f"{t}* __{self.label}__:\n"
        for _, p in self._parameters.items():
            s += f"{t}  * {p}\n"
        for _, g in self.items():
            s += f"{g.__str__()}"
        return s

    def __str__(self):
        return self.markdown()
