class GlobalAnalysis(object):
    """
    Represents a global analysis job.
    """
    def __init__(self, datasets, model):
        self._datasets = datasets
        self._model = model

    def fit(self):
        raise NotImplementedError

    def result(self):
        if self._result is None:
            raise Exception("Fitting hasn't run yet.")
        else:
            return self._result
