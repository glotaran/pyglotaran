from collections import OrderedDict

from lmfit import Parameters

from .parameter import Parameter


class ParameterLeaf(OrderedDict):
    def __init__(self, label):
        self._label = label
        self._parameters = OrderedDict()
        self._fit = True
        super(ParameterLeaf, self).__init__()

    def add_parameter(self, parameter):
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
                p.vary = False
            self._parameters[p.label] = p

    def add_leaf(self, leaf):
        if not isinstance(leaf, ParameterLeaf):
            raise TypeError("Leave must be glotaran.model.ParameterLeaf")
        if not self.fit:
            leaf.fit = False
        self[leaf.label] = leaf

    @property
    def fit(self):
        return self._fit

    @fit.setter
    def fit(self, value):
        for p in self.all_leaf():
            p.vary = value
        self._fit = value
        for _, l in self.items():
            l.fit = value

    @property
    def label(self):
        return self._label

    def leafs(self):
        for leaf in self:
            for l in leaf.leafs():
                yield l

    def get(self, label):
        path = label.split(".")
        label = path.pop()
        leaf = self
        for l in path:
            leaf = leaf[l]
        try:
            return leaf._parameters[label]
        except:
            raise Exception("Cannot find parameter "
                            "{}".format(".".join(path)+"."+label))

    def get_by_index(self, idx):
        return [i for _, i in self._parameters.items()][idx-1]

    def all_leaf(self):
        for _, p in self._parameters.items():
            yield p

    def all(self):
        for p in self.all_leaf():
            yield p
        for l in self:
            for p in l.all():
                yield p

    def all_with_label(self, root):
        root = "{}_{}".format(root, self.label) if root is not None else \
            self.label
        for label, p in self._parameters.items():
            yield ("{}_{}".format(root, label), p)
        for _, l in self.items():
            for (lbl, p) in l.all_with_label(root):
                yield (lbl, p)

    def as_parameters_dict(self):
        params = Parameters()
        for (label, p) in self.all_with_label(None):
            p.name = label
            params.add(p)
        return params
