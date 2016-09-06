try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from dataset import Dataset


class Datasets(OrderedDict):
    """
    A dictionary of all datasets to be fitted.
    """
    def __init__(self, datasets):
        super(Datasets, self)
        if not isinstance(datasets, list):
            datasets = [datasets]
        if any(not isinstance(dataset, Dataset) for dataset in datasets):
            raise TypeError
        for dataset in datasets:
            self.add(dataset)

    def add(self, dataset):
        """
        Add a dataset.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError
        if dataset.label() in self:
            raise Exception("Labels must be unique.")
        self.__setitem__(dataset.label(), dataset)
