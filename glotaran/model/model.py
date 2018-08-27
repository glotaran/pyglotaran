"""Glotaran Model"""


from abc import ABC, abstractmethod
from typing import List, Type, Dict, Generator
from collections import OrderedDict
import numpy as np

from glotaran.fitmodel import FitModel, Matrix, Result

from .dataset import Dataset
from .dataset_descriptor import DatasetDescriptor
from .decorators import glotaran_model
from .initial_concentration import InitialConcentration
from .megacomplex import Megacomplex
from .parameter_group import ParameterGroup

ROOT_BLOCK_LABEL = "p"


@glotaran_model("base_model",
    attributes={'initial_concentration': InitialConcentration,
                'megacomplex': Megacomplex}
)
class Model(ABC):
    """Model represents a global analysis model."""

    compartments: List[str] = []
    _datasets = OrderedDict()

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=attribute-defined-outside-init
    # Models are complex.

    @classmethod
    def from_dict(cls, model_dict):
        print(cls)

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

    @abstractmethod
    def calculated_matrix(self) -> Type[Matrix]:
        """Returns the calculated matrix.

        Returns
        -------

        matrix : type(fitmodel.Matrix)
            Calculated Matrix
        """
        raise NotImplementedError

    @abstractmethod
    def estimated_matrix(self) -> Type[Matrix]:
        """Returns the estimated matrix.

        Returns
        -------

        matrix : type(fitmodel.Matrix)
            Estimated Matrix
        """
        raise NotImplementedError

    @abstractmethod
    def dataset_class(self) -> Type[DatasetDescriptor]:
        """Returns an implementation for model.DatasetDescriptor.

        Returns
        -------

        descriptor : type(model.DatasetDescriptor)
            Implementation of model.DatasetDescriptor
        """
        raise NotImplementedError

    @abstractmethod
    def fit_model_class(self) -> Type[FitModel]:
        """Returns an implementation for fitmodel.FitModel.

        Returns
        -------

        fitmodel : type(fitmodel.FitModel)
            Implementation of fitmodel.FitModel
        """
        raise NotImplementedError

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
        return self.fit_model().fit(parameter.as_parameter_dict(only_fit=True),
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
        data = self.dataset_class()(dataset)
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

    def fit_model(self) -> FitModel:
        """Returns an instance of the models fitmodel.FitModel implementation.

        Returns
        -------

        fitmodel : fitmodel.FitModel
        """
        return self.fit_model_class()(self)

    def data(self) -> Generator[DatasetDescriptor, None, None]:
        """Gets all datasets as a generator.

        Returns
        -------

        datasets : generator(DatasetDescriptor)
        """
        for _, dataset in self.datasets.items():
            yield dataset.data

    def list_datasets(self) -> List[str]:
        """Returns a list of all dataset labels

        Returns
        -------

        datasets : list(str)
        """
        return [label for label in self.datasets]

    @property
    def datasets(self) -> Dict[str, DatasetDescriptor]:
        """A dictonary of all datasets"""
        return self._datasets

    @datasets.setter
    def datasets(self, value):
        if not isinstance(value, OrderedDict):
            raise TypeError("Datasets must be Ordered.")
        if any(not issubclass(type(value[val]),
                              DatasetDescriptor) for val in value):
            raise TypeError("Dataset must be subclass of 'DatasetDescriptor'")
        self._datasets = value

    def add_dataset(self, dataset: DatasetDescriptor):
        """Adds a DatasetDescriptor to the model

        Parameters
        ----------
        dataset : DatasetDescriptor


        """
        if not issubclass(type(dataset), DatasetDescriptor):
            raise TypeError("Dataset must be subclass of 'DatasetDescriptor'")
        self.datasets[dataset.label] = dataset

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
        return self.datasets[label].dataset

    def set_dataset(self, label: str, dataset: Dataset):
        """ Sets the dataset of a DatasetDescriptor

        Parameters
        ----------
        label : str
            Label of the DatasetDescriptor

        dataset : Dataset
            The Dataset


        """
        self.datasets[label].dataset = dataset

    def __str__(self):
        string = "# Model\n\n"
        string += "_Type_: {}\n\n".format(self.glotaran_model_type)

        if self.datasets is not None:
            string += "\n## Datasets\n\n"

            for data in self._datasets:
                string += "{}\n".format(self._datasets[data])

        if self.compartments is not None:
            string += "\n## Compartments\n\n"

            for comp in self.compartments:
                string += f"* {comp}\n"

        string += "\n## Megacomplexes\n\n"

        for mcp in self.megacomplex:
            string += "{}\n".format(self.megacomplex[mcp])

        string += "\n## Initital Concentrations\n\n"

        for i in self.initial_concentration:
            string += "{}\n".format(self.initial_concentration[i])

        return string
