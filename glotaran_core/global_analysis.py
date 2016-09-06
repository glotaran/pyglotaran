from datasets import Datasets

class GlobalAnalysis(object):
    """
    Represents a global analysis job.

    Parameter: A Datasets object representing the datasets to be fitted and a
    Models object representing the fitmodels.

    Datasets and models are linked by labels.
    """
    def __init__(self, datasets, models):
        

        if not isinstance(datasets, Datasets):
            raise TypeError


        self._datasets = datasets
        self._model = model

    def fit(self):
        raise NotImplementedError

    def result(self):
        if self._result is None:
            raise Exception("Fitting hasn't run yet.")
        else:
            return self._result
