"""Glotaran Model"""


from abc import ABC, abstractmethod
from typing import List, Type, Dict, Generator
from collections import OrderedDict
import numpy as np

from .dataset import Dataset
from .dataset_descriptor import DatasetDescriptor
from .initial_concentration import InitialConcentration
from .megacomplex import Megacomplex
from .parameter_group import ParameterGroup

from glotaran.fitmodel import FitModel, Matrix, Result

ROOT_BLOCK_LABEL = "p"


class Model(ABC):
    """Model represents a global analysis model."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=attribute-defined-outside-init
    # Models are complex.
    def __init__(self):
        """ """
        self._compartments = None
        self._datasets = OrderedDict()
        self._initial_concentrations = None
        self._megacomplexes = {}
        self._parameter = None

    @abstractmethod
    def type_string(self) -> str:
        """Returns a human readable string identifying the type of the model.

        Returns
        -------

        type : str
            Type of the Model

        """
        raise NotImplementedError

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

    def fit(self, *args, nnls=False, **kwargs) -> Type[Result]:
        """ Fits the model and returns the result.

        Parameters
        ----------
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
        return self.fit_model().fit(nnls, *args, **kwargs)

    def simulate(self,
                 dataset: str,
                 axis: Dict[str, np.array],
                 parameter=None,
                 noise=False,
                 noise_std_dev=1.0,
                 ):
        """Simulates the model.

        Parameters
        ----------
        dataset : str
            Label of the dataset to simulate

        axis : dict(str, np.array)
            A dictory with axis
        parameter :
            (Default value = None)
        noise :
            (Default value = False)
        noise_std_dev :
            (Default value = 1.0)

        """
        data = self.dataset_class()(dataset)
        sim_parameter = self.parameter.as_parameters_dict().copy()
        if parameter is not None:
            for k, val in parameter.items():
                k = "p_" + k.replace(".", "_")
                sim_parameter[k].value = val
        for label, val in axis.items():
            data.set_axis(label, val)
        self.datasets[dataset].dataset = data

        kwargs = {}
        kwargs['dataset'] = dataset
        kwargs['noise'] = noise
        kwargs['noise_std_dev'] = noise_std_dev
        data = self.fit_model().eval(sim_parameter, **kwargs)
        self.get_dataset(dataset).set(data)

    def concentrations(self, dataset: str) -> np.ndarray:
        """Returns the precited concentrations for a dataset.

        Parameters
        ----------
        dataset : str
            Label of the dataset

        Returns
        -------
        concentrations : numpy.ndarray
        """
        parameter = self.parameter.as_parameters_dict().copy()
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

    @property
    def compartments(self) -> List[str]:
        """A list of compartment labels."""
        return self._compartments

    @compartments.setter
    def compartments(self, value):
        self._compartments = value

    @property
    def parameter(self) -> ParameterGroup:
        """The model parameters."""
        return self._parameter

    @parameter.setter
    def parameter(self, val):
        if not isinstance(val, ParameterGroup):
            raise TypeError
        self._parameter = val

    @property
    def megacomplexes(self) -> Dict[str, Megacomplex]:
        """A dictonary of megacomplexes."""
        return self._megacomplexes

    @megacomplexes.setter
    def megacomplexes(self, value):
        if not isinstance(value, dict):
            raise TypeError("Megacomplexes must be dict.")
        if any(not issubclass(type(value[val]), Megacomplex) for val in value):
            raise TypeError("Megacomplexes must be subclass of 'Megacomplex'")
        self._megacomplexes = value

    def add_megacomplex(self, megacomplex: Megacomplex):
        """Adds a megacomplex to the model.

        Parameters
        ----------
        megacomplex : Megacomplex

        """
        if not issubclass(type(megacomplex), Megacomplex):
            raise TypeError("Megacomplexes must be subclass of 'Megacomplex'")
        if self.megacomplexes is not None:
            if megacomplex.label in self.megacomplexes:
                raise Exception("Megacomplex labels must be unique")
            self.megacomplexes[megacomplex.label] = megacomplex
        else:
            self.megacomplexes = {megacomplex.label: megacomplex}

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

    @property
    def initial_concentrations(self) -> Dict[str, InitialConcentration]:
        """A Dictoinary of the initial concentrations."""
        return self._initial_concentrations

    @initial_concentrations.setter
    def initial_concentrations(self, value):
        if not isinstance(value, dict):
            raise TypeError("Initial concentrations must be dict.")
        if any(not isinstance(value[val],
                              InitialConcentration) for val in value):
            raise TypeError("Initial concentrations must be instance of"
                            " 'InitialConcentration'")
        self._initial_concentrations = value

    def add_initial_concentration(self, initial_concentration: InitialConcentration):
        """Adds an initial concentration to the model.

        Parameters
        ----------
        initial_concentration : InitialConcentration


        """
        if not isinstance(initial_concentration, InitialConcentration):
            raise TypeError("Initial concentrations must be instance of"
                            " 'InitialConcentration'")
        if self.initial_concentrations is not None:
            self.initial_concentrations[initial_concentration.label] =\
                initial_concentration
        else:
            self.initial_concentrations = {initial_concentration.label:
                                           initial_concentration}

    def __str__(self):
        string = "# Model\n\n"
        string += "_Type_: {}\n\n".format(self.type_string())

        string += "## Parameter\n{}\n".format(self.parameter)

        if self.datasets is not None:
            string += "\n## Datasets\n\n"

            for data in self._datasets:
                string += "{}\n".format(self._datasets[data])

        if self.compartments is not None:
            string += "\n## Compartments\n\n"

            for c in self.compartments:
                string += f"* {c}\n"

        string += "\n## Megacomplexes\n\n"

        for mcp in self._megacomplexes:
            string += "{}\n".format(self._megacomplexes[mcp])

        if self.initial_concentrations is not None:
            string += "\n## Initital Concentrations\n\n"

            for i in self._initial_concentrations:
                string += "{}\n".format(self._initial_concentrations[i])

        return string
