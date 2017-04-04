from .parameter import Parameter


class ParameterBlock(object):
    def __init__(self, label, fit=True):
        self.label = label
        self.fit = fit
        self.parameter = []
        self.sub_blocks = {}
        self._root = None

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
            if self.root is not None:
                p.index = "{}.{}".format(self.index(), p.index)
            self._parameter.append(p)

    def add_sub_block(self, block):
        block.root = self.index()
        self._sub_blocks[block.label] = block

    @property
    def sub_blocks(self):
        return self._sub_blocks

    @sub_blocks.setter
    def sub_blocks(self, value):
        self._sub_blocks = value

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value
        i = 0
        for p in self.parameter:
            i += 1
            p.index = "{}.{}".format(self.index(), i)
        for _, block in self.sub_blocks.items():
            block.root = self.index()

    def index(self):
        if self.root is None:
            return self.label
        else:
            return "{}.{}".format(self.root, self.label)

    def all_parameter(self):
        for p in self.parameter:
            yield p
        for _, block in self.sub_blocks.items():
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

    def _str__(self):
        s = "Label: {}\n".format(self.label)
        s += "Parameter: {}\n".format(self.parameter)
        s += "Fit: {}\n".format(self.fit)
        if self.sub_blocks is not None:
            for block in self.sub_blocks:
                bs = sum(["\t"+l for l in block.__str__().splitlines()])
                s += bs + "\n"
        return s


def _check_param_list(parameter):
        if not isinstance(parameter, list):
            parameter = [parameter]

        if not all(isinstance(p, Parameter) for p in parameter):
            raise TypeError("Parameters must be instance of"
                            " glotaran.model.Parameter")
        return parameter
