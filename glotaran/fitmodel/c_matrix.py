
class CMatrix(object):
    def __init__(self, buffer, x, dataset, model):
        self.x = x
        self.dataset = dataset
        self.buffer = buffer

    def calculate(self, parameter):
        raise NotImplementedError
