from datasets import Datasets
from dataset import Dataset
from models import Models
from model import Model


class GlobalAnalysis(object):
    """
    Represents a global analysis job.

    Parameter: A Dataset or Datasets object representing the dataset(s) to be
    fitted and a Model or Models object representing the fitmodel(s).

    Datasets and models are linked by labels.
    """
    def __init__(self, datasets, models):

        if isinstance(datasets, Dataset):
            datasets = Datasets(datasets)
        if not isinstance(datasets, Datasets):
            raise TypeError

        if isinstance(models, Model):
            models = Models(models)
        if not isinstance(models, Models):
            raise TypeError
        self._datasets = datasets
        self._models = models

    def fit(self):
        raise NotImplementedError

    def result(self):
        if self._result is None:
            raise Exception("Fitting hasn't run yet.")
        else:
            return self._result
