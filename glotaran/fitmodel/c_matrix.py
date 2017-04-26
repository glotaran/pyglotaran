
class CMatrix(object):
    def __init__(self, x, dataset, model):
        self.x = x
        self.dataset = dataset
        #  self.buffer = buffer

    def calculate(self, parameter):
        raise NotImplementedError

    def compartment_order(self):
        raise NotImplementedError

    def shape(self):
        raise NotImplementedError
