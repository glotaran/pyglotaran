

class GlobalAnalysis(object):
    """
    Represents a global analysis job.
    """
    def __init__(self, model):
        if not issubclass(type(model), Model):
            raise TypeError
        self.model = model
        self._result = None

    def fit(self):
        raise NotImplementedError

    def result(self):
        if self._result is None:
            raise Exception("Fitting hasn't run yet.")
        else:
            return self._result
