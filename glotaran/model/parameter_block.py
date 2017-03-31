class ParameterBlock(object):
    def __init__(self, label, parameter, sub_blocks=None, fit=True):
        self.label = label
        self.parameter = parameter
        self.sub_blocks = sub_blocks
        self.fit = fit

    def add_sub_block(self, block):
        if self.sub_blocks is None:
            self.sub_blocks = {block.label: block}
        else:
            self.sub_blocks[block.label] = block

    def __str__(self):
        s = "Label: {}\n".format(self.label)
        s += "Parameter: {}\n".format(self.parameter)
        s += "Fit: {}\n".format(self.fit)
        if self.sub_blocks is not None:
            for block in self.sub_blocks:
                bs = sum(["\t"+l for l in block.__str__().splitlines()])
                s += bs + "\n"
