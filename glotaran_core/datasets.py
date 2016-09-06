try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from dataset import Dataset


class Datasets(OrderedDict):
    """
    A dictionary of all datasets to be fitted. Names must agree with names in
    model spec.
    """
    def __init__(self):
        super(Datasets, self)

    def add(self, dataset):
        """
        Add a dataset.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError

        self.__setitem__(dataset.label(), dataset)
