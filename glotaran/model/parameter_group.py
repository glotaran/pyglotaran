"""Glotaran Parameter Group"""

from typing import Generator, Tuple
from collections import OrderedDict

from lmfit import Parameters

from .parameter import Parameter


class ParameterGroup(OrderedDict):
    """Represents are group of parameters. Can contain other groups, creating a
    tree-like hirachy."""
    def __init__(self, label: str):
        self._label = label
        self._parameters = OrderedDict()
        self._fit = True
        self._root = None
        super(ParameterGroup, self).__init__()

    @classmethod
    def from_parameter_dict(cls, parameter: Parameters):
        """Creates a parameter group from an lmfit.Parameters dictionary

        Parameters
        ----------
        parameter : lmfit.Parameters
            lmfit.Parameters dictionary
        """

        root = cls("p")

        for lbl, param in parameter.items():
            lbl = lbl.split("_")
            if len(lbl) is 2:
                # it is a root param
                param = Parameter.from_parameter(lbl.pop(), param)
                root.add_parameter(param)
                continue

            # remove root
            lbl.pop(0)

            top = root
            while len(lbl) is not 0:
                group = lbl.pop(0)
                if group in top:
                    if len(lbl) is 1:
                        param = Parameter.from_parameter(lbl.pop(), param)
                        top[group].add_parameter(param)
                    else:
                        top = top[group]
                else:
                    group = ParameterGroup(group)
                    top.add_group(group)
                    if len(lbl) is 1:
                        param = Parameter.from_parameter(lbl.pop(), param)
                        group.add_parameter(param)
                    else:
                        top = group
        return root

    def add_parameter(self, parameter: Parameter):
        """

        Parameters
        ----------
        parameter : Parameter

        """
        if not isinstance(parameter, list):
            parameter = [parameter]
        if any(not isinstance(p, Parameter) for p in parameter):
            raise TypeError("Parameter must be  instance of"
                            " glotaran.model.Parameter")
        for p in parameter:
            p.index = len(self._parameters) + 1
            if p.label is None:
                p.label = "{}".format(p.index)
            if not self.fit:
                p.fit = False
            self._parameters[p.label] = p

    def add_group(self, group: 'ParameterGroup'):
        """

        Parameters
        ----------
        group : ParameterGroup

        """
        if not isinstance(group, ParameterGroup):
            raise TypeError("Leave must be glotaran.model.ParameterGroup")
        if not self.fit:
            group.fit = False
        group.set_root(self)
        self[group.label] = group

    def set_root(self, root):
        self._root = root

    def get_nr_roots(self):
        n = 0
        root = self._root
        while root is not None:
            n += 1
            root = root._root
        return n

    @property
    def fit(self):
        """Indicates if the group should be filtered out of the fitting process"""
        return self._fit

    @fit.setter
    def fit(self, value):
        for p in self.all_group():
            p.fit = value
        self._fit = value
        for _, l in self.items():
            l.fit = value

    @property
    def label(self):
        """Label of the group """
        return self._label

    def groups(self) -> Generator['ParameterGroup', None, None]:
        """Generator over all groups and their subgroups"""
        for group in self:
            for l in group.groups():
                yield l

    def get(self, label: str) -> Parameter:
        """Gets a parameter by it label.

        Parameters
        ----------
        label : str
            Label of the Parameter to get.


        Returns
        -------
        parameter : Parameter

        """

        # sometimes the spec parser delivers the labels as int
        label = str(label)

        path = label.split(".")
        label = path.pop()

        group = self
        for l in path:
            group = group[l]
        try:
            return group._parameters[label]
        except KeyError:
            raise Exception("Cannot find parameter "
                            "{}".format(".".join(path)+"."+label))

    def get_by_index(self, idx: int) -> Parameter:
        """ Gets a parameter by its index. Only works for unlabeled parameters
        in the root group.

        Parameters
        ----------
        idx : int
            Index of the parameter.

        Returns
        -------
        parameter : Parameter
        """
        return [i for _, i in self._parameters.items()][idx-1]

    def all_group(self) -> Generator[Parameter, None, None]:
        """Generator returning all Parameter within the group, but not in subgroups"""
        for _, p in self._parameters.items():
            yield p

    def all(self) -> Generator[Parameter, None, None]:
        """Generator returning all parameters within the group and in subgroups"""
        for p in self.all_group():
            yield p
        for l in self:
            for p in self[l].all():
                yield p

    def all_with_label(self, root) -> Generator[Tuple[str, Parameter], None,
                                                None]:
        """ Same as all, but returns the labels relative to the given root
        group.
        Parameters
        ----------
        root : label of the root group


        """
        root = "{}_{}".format(root, self.label) if root is not None else \
            self.label
        for label, p in self._parameters.items():
            yield ("{}_{}".format(root, label), p)
        for _, l in self.items():
            for (lbl, p) in l.all_with_label(root):
                yield (lbl, p)

    def as_parameters_dict(self, only_fit=False) -> Parameters:
        """
        Creates a lmfit.Parameters dict.

        Parameters
        ----------
        only_fit : bool
            (Default value = False)
            if True, all parameters with fit = False will be filtered

        Returns
        -------
        Parameters : lmfit.Parameters
        """
        params = Parameters()
        for (label, p) in self.all_with_label(None):
            p.name = label
            if not only_fit or p.fit:
                params.add(p)
        return params

    def __str__(self):
        t = "".join(["  " for _ in range(self.get_nr_roots())])
        s = ""
        if self.label is not "p":
            s += f"{t}* __{self.label}__:\n"
        for _, p in self._parameters.items():
            s += f"{t}  * {p}\n"
        for _, g in self.items():
            s += f"{g.__str__()}"
        return s
