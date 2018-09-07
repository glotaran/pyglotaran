"""Glotaran Model"""


from abc import ABC, abstractmethod
from typing import List, Type, Dict, Generator
from collections import OrderedDict
import numpy as np
import inspect

from glotaran.math.fitresult import Result
from glotaran.math.simulation import simulate
from .dataset import Dataset
from .dataset_descriptor import DatasetDescriptor
from .initial_concentration import InitialConcentration
from .megacomplex import Megacomplex
from .parameter_group import ParameterGroup


class Model:
    """Model represents a global analysis model."""

    compartment: List[str]

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=no-member
    # Models are complex.

    def __init__(self):
        self.compartment = []

    @classmethod
    def from_dict(cls, model_dict):

        model = cls()
        if 'compartment' in model_dict:
            model.compartment = model_dict['compartment']
            del model_dict['compartment']

        for name, attribute in list(model_dict.items()):
            if hasattr(model, f'set_{name}'):
                set = getattr(model, f'set_{name}')
                item_cls = set.__func__.__annotations__['item']
                for label, item in attribute.items():
                    is_typed = hasattr(item_cls, "_glotaran_model_item_typed")
                    if isinstance(item, dict):
                        if is_typed:
                            if 'type' not in item:
                                raise Exception(f"Missing type for attribute '{name}'")
                            item_type = item['type']

                            if item_type not in item_cls._glotaran_model_item_types:
                                raise Exception(f"Unknown type '{item_type}' "
                                                f"for attribute '{name}'")
                            item_cls = \
                                item_cls._glotaran_model_item_types[item_type]
                        item['label'] = label
                        set(label, item_cls.from_dict(item))
                    elif isinstance(item, list):
                        if is_typed:
                            if len(item) < 2 and len(item) is not 1:
                                raise Exception(f"Missing type for attribute '{name}'")
                            item_type = item[1] if len(item) is not 1 and \
                                    hasattr(item_cls,'label') else item[0]

                            if item_type not in item_cls._glotaran_model_item_types:
                                raise Exception(f"Unknown type '{item_type}' "
                                                f"for attribute '{name}'")
                            item_cls = \
                                item_cls._glotaran_model_item_types[item_type]
                        item = [label] + item
                        set(label, item_cls.from_list(item))
                del model_dict[name]

        return model

    def simulate(self, dataset: str, parameter: ParameterGroup, axis: Dict[str, np.ndarray]):
        return simulate(self, parameter, dataset, axis)

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


    def get_data(self, label: str) -> Dataset:
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

    def set_data(self, label: str, dataset: Dataset):
        """ Sets the dataset of a DatasetDescriptor

        Parameters
        ----------
        label : str
            Label of the DatasetDescriptor

        dataset : Dataset
            The Dataset


        """
        self.dataset[label].dataset = dataset

    def errors(self):
        attrs = getattr(self, '_glotaran_model_attributes')

        errors = []

        for attr in attrs:
            for _, item in getattr(self, attr).items():
                item.validate_model(self, errors=errors)

        return errors

    def valid(self):
        return len(self.errors()) is 0

    def errors_parameter(self, parameter):
        attrs = getattr(self, '_glotaran_model_attributes')

        errors = []

        for attr in attrs:
            for _, item in getattr(self, attr).items():
                item.validate_parameter(self, parameter, errors=errors)

        return errors

    def valid_parameter(self, parameter):
        return len(self.errors_parameter(parameter)) is 0

    def __str__(self):
        attrs = getattr(self, '_glotaran_model_attributes')
        string = "# Model\n\n"
        string += f"_Type_: {self.model_type}\n\n"

        for attr in attrs:
            string += f"## {attr}\n"

            for label, item in getattr(self, attr).items():
                string += f'{item}\n'
        return string
