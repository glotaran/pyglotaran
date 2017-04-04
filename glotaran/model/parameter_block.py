from .parameter import Parameter


class ParameterBlock(object):
    def __init__(self, label, parameter=[], root=None, sub_blocks={},
                 fit=True):
        self.label = label
        self.parameter = parameter
        self.sub_blocks = sub_blocks
        self.fit = fit
        self.root = root

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, parameter):
        parameter = _check_param_list(parameter)
        self._parameter = parameter

    def add_parameter(self, parameter):
        parameter = _check_param_list(parameter)
        for p in parameter:
            p.index = len(self._parameter)
            self._parameter.append(p)

    def add_sub_block(self, block):
        block.root = self.index()

        if self.sub_blocks is None:
            self.sub_blocks = {block.label: block}
        else:
            self.sub_blocks[block.label] = block

    def index(self):
        if self.root is None:
            return self.label
        else:
            return "{}.{}".format(self.root, self.label)

    def all_parameter(self):
        for p in self.parameter:
            yield p
        for block in self.sub_blocks:
            for p in block.all_parameter():
                yield p

    def as_lmfit_parameter(self, only_fit=False):
        if only_fit and not self.fit:
            return
        for p in self.parameter:
            yield p.as_lmfit_parameter()
        for block in self.sub_blocks:
            for p in block.as_lmfit_parameter(only_fit=only_fit):
                yield p

    def __str__(self):
        s = "Label: {}\n".format(self.label)
        s += "Parameter: {}\n".format(self.parameter)
        s += "Fit: {}\n".format(self.fit)
        if self.sub_blocks is not None:
            for block in self.sub_blocks:
                bs = sum(["\t"+l for l in block.__str__().splitlines()])
                s += bs + "\n"


def _check_param_list(parameter):
        if not isinstance(parameter, list):
            parameter = [parameter]

        if not all(isinstance(p, Parameter) for p in parameter):
            raise TypeError("Parameters must be instance of"
                            " glotaran.model.Parameter")
        return parameter
