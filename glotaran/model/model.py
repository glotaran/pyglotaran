"""Glotaran Model"""


from abc import ABC, abstractmethod
from typing import List, Type, Dict, Generator
from collections import OrderedDict
import numpy as np
import inspect

from glotaran.fitmodel import FitModel, Matrix, Result

from .dataset import Dataset
from .dataset_descriptor import DatasetDescriptor
from .initial_concentration import InitialConcentration
from .megacomplex import Megacomplex
from .parameter_group import ParameterGroup


class Model:
    """Model represents a global analysis model."""

    compartments: List[str]

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=no-member
    # Models are complex.

    def __init__(self):
        self.compartments = []

    @classmethod
    def from_dict(cls, model_dict):

        model = cls()
        model.compartments = model_dict['compartments']
        del model_dict['compartments']

        for name, attribute in list(model_dict.items()):
            if hasattr(model, f'set_{name}'):
                set = getattr(model, f'set_{name}')
                item_cls = set.__func__.__annotations__['item']
                for label, item in attribute.items():
                    if isinstance(item, dict):
                        item['label'] = label
                        set(label, item_cls.from_dict(item))
                    elif isinstance(item, list):
                        item = [label] + item
                        set(label, item_cls.from_list(item))
                del model_dict[name]

        return model

    def fit(self, parameter: ParameterGroup, *args, nnls=False, **kwargs) -> Type[Result]:
        """ Fits the model and returns the result.

        Parameters
        ----------
        parameter : ParameterGroup
        nnls :
            (Default value = False)
        *args :

        **kwargs :


        Returns
        -------
        result : type(fitmodel.Result)

        """
        if any([dset.dataset is None for _, dset in self.datasets.items()]):
            raise Exception("Model datasets not initialized")
        return self.fitmodel().fit(parameter.as_parameter_dict(only_fit=True),
                                   *args,
                                   nnls=nnls,
                                   **kwargs)

    def simulate(self,
                 parameter: ParameterGroup,
                 dataset: str,
                 axis: Dict[str, np.ndarray],
                 noise=False,
                 noise_std_dev=1.0,
                 ):
        """Simulates the model.

        Parameters
        ----------
        parameter : ParameterGroup
            The parameters for the simulation.
        dataset : str
            Label of the dataset to simulate

        axis : dict(str, np.ndarray)
            A dictory with axis
        noise :
            (Default value = False)
        noise_std_dev :
            (Default value = 1.0)

        """
        data = self.dataset_type(dataset)
        parameter = parameter.as_parameter_dict()
        for label, val in axis.items():
            data.set_axis(label, val)
        self.datasets[dataset].dataset = data

        kwargs = {}
        kwargs['dataset'] = dataset
        kwargs['noise'] = noise
        kwargs['noise_std_dev'] = noise_std_dev
        data = self.fit_model().eval(parameter, **kwargs)
        self.get_dataset(dataset).set(data)

    def concentrations(self, parameter: ParameterGroup, dataset: str) -> np.ndarray:
        """Returns the precited concentrations for a dataset.

        Parameters
        ----------
        dataset : str
            Label of the dataset

        Returns
        -------
        concentrations : numpy.ndarray
        """
        parameter = parameter.as_parameter_dict().copy()
        kwargs = {}
        kwargs['dataset'] = dataset
        return self.fit_model().c_matrix(parameter, **kwargs)

    def fitmodel(self) -> FitModel:
        """Returns an instance of the models fitmodel.FitModel implementation.

        Returns
        -------

        fitmodel : fitmodel.FitModel
        """
        return self.fitmodel_type(self)

    def data(self) -> Generator[DatasetDescriptor, None, None]:
        """Gets all datasets as a generator.

        Returns
        -------

        datasets : generator(DatasetDescriptor)
        """
        for _, dataset in self.dataset.items():
            yield dataset.data

    def list_datasets(self) -> List[str]:
        """Returns a list of all dataset labels

        Returns
        -------

        datasets : list(str)
        """
        return [label for label in self.dataset]


    def get_dataset(self, label: str) -> Dataset:
        """ Sets the dataset of a DatasetDescriptor

        Parameters
        ----------
        label : str
            Label of the DatasetDescriptor

        Returns
        -------
        dataset : Dataset
            The Dataset


        """
        return self.dataset[label].dataset

    def set_dataset(self, label: str, dataset: Dataset):
        """ Sets the dataset of a DatasetDescriptor

        Parameters
        ----------
        label : str
            Label of the DatasetDescriptor

        dataset : Dataset
            The Dataset


        """
        self.dataset[label].dataset = dataset


