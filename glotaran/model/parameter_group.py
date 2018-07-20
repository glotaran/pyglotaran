from collections import OrderedDict

from lmfit import Parameters

from .parameter import Parameter


class ParameterGroup(OrderedDict):
    """Represents are group of parameters. Can contain other groups, creating a
    tree-like hirachy."""
    def __init__(self, label):
        self._label = label
        self._parameters = OrderedDict()
        self._fit = True
        super(ParameterGroup, self).__init__()

    def add_parameter(self, parameter):
        """

        Parameters
        ----------
        parameter : instance of Parameter


        Returns
        -------

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

    def add_group(self, group):
        """

        Parameters
        ----------
        group : instance of ParameterGroup


        Returns
        -------

        """
        if not isinstance(group, ParameterGroup):
            raise TypeError("Leave must be glotaran.model.ParameterGroup")
        if not self.fit:
            group.fit = False
        self[group.label] = group

    @property
    def fit(self):
        """Indicates if the group should be filtered out of the fitting process"""
        return self._fit

    @fit.setter
    def fit(self, value):
        """

        Parameters
        ----------
        value : True or False


        Returns
        -------

        """
        for p in self.all_group():
            p.fit = value
        self._fit = value
        for _, l in self.items():
            l.fit = value

    @property
    def label(self):
        """Label of the group """
        return self._label

    def groups(self):
        """Generator over all groups their subgroups"""
        for group in self:
            for l in group.groups():
                yield l

    def get(self, label):
        """

        Parameters
        ----------
        label : label of the Parameter to get.


        Returns
        -------
        parameter

        """
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

    def get_by_index(self, idx):
        """

        Parameters
        ----------
        idx : index of the Parameter.


        Returns
        -------
        parameter
        """
        return [i for _, i in self._parameters.items()][idx-1]

    def all_group(self):
        """Generates all parameter within the group, but not in subgroups"""
        for _, p in self._parameters.items():
            yield p

    def all(self):
        """Generates all parameters within the group and in subgroups"""
        for p in self.all_group():
            yield p
        for l in self:
            for p in l.all():
                yield p

    def all_with_label(self, root):
        """
        Same as all, but prepends tree hirachy to the parameter labels.
        Parameters
        ----------
        root : label of the root group


        Returns
        -------

        """
        root = "{}_{}".format(root, self.label) if root is not None else \
            self.label
        for label, p in self._parameters.items():
            yield ("{}_{}".format(root, label), p)
        for _, l in self.items():
            for (lbl, p) in l.all_with_label(root):
                yield (lbl, p)

    def as_parameters_dict(self, only_fit=False):
        """
        Creates a lmfit.Parameters dict.
        Parameters
        ----------
        only_fit : if True, all parameters with fit = False will be filtered
        out
             (Default value = False)

        Returns
        -------
        Parameters : instance of lmfit.Parameters
        """
        params = Parameters()
        for (label, p) in self.all_with_label(None):
            p.name = label
            if not only_fit or p.fit:
                params.add(p)
        return params

    def __str__(self):
        s = "Label\tValue\tMin\tMax\tFix\n"
        for _, p in self.as_parameters_dict().items():
            s += "\t{}\n".format(p)
        return s
