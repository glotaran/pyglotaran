"""This package contains glotarans base model."""


from typing import List, Dict
import numpy as np

from glotaran.analysis.fitresult import FitResult
from glotaran.analysis.fitting import fit
from glotaran.analysis.simulation import simulate

from glotaran.model.dataset import Dataset
from .parameter_group import ParameterGroup


class BaseModel:
    """Base Model contains basic functions for model.

    The basemodel already has the compartment attribute, which all model
    implementations will need.
    """

    compartment: List[str]
    """A list of compartments."""

    def __init__(self):
        self.compartment = []

    @classmethod
    def from_dict(cls, model_dict: Dict):
        """from_dict creates a model from a dictionary.

        Parameters
        ----------
        model_dict : dict
            Dictionary containing the model.

        Returns
        -------
        model : The parsed model.
        """

        model = cls()

        # first get the compartments.
        if 'compartment' in model_dict:
            model.compartment = model_dict['compartment']
            del model_dict['compartment']

        # iterate over items
        for name, attribute in list(model_dict.items()):

            # we determine if we the item is known by the model by looking for
            # a setter with same name.
            if hasattr(model, f'set_{name}'):

                # get the set function
                set = getattr(model, f'set_{name}')

                # we retrieve the actual class from the signature
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
                                hasattr(item_cls, 'label') else item[0]

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

    def fit(self,
            parameter: ParameterGroup,
            data: Dict[str, Dataset],
            verbose: int = 2,
            max_fnev: int = None,) -> FitResult:
        """fit performs a fit of the model.

        Parameters
        ----------
        parameter : ParameterGroup
            The initial fit parameter.
        data : dict(str, glotaran.model.dataset.Dataset)
            A dictionary of dataset labels and Datasets.
        verbose : int
            (optional default=2)
            Set 0 for no log output, 1 for only result and 2 for full verbosity
        max_fnev : int
            (optional default=None)
            The maximum number of function evaluations, default None.

        Returns
        -------
        result: FitResult
            The result of the fit.
        """
        return fit(self, parameter, data, verbose=verbose, max_fnev=max_fnev)

    def calculated_matrix(self,
                          dataset: str,
                          parameter: ParameterGroup,
                          index: any,
                          axis: np.ndarray) -> np.ndarray:
        """calculated_matrix returns the calculated matrix for a dataset.

        Parameters
        ----------
        dataset : str
            Label of the dataset
        parameter : ParameterGroup
            The parameter for the prediction
        index : int
            The index on the estimated axis,
        axis : numpy.ndarray
            The calculated axis.

        Returns
        -------
        calculated_matrix : numpy.ndarray
        """
        filled_dataset = self.dataset[dataset].fill(self, parameter)
        return self.calculated_matrix(filled_dataset,
                                      self.compartment,
                                      index,
                                      axis)

    def errors(self) -> List[str]:
        """errors returns a list of errors in the model.

        Returns
        -------
        errors : List[str]
        """
        attrs = getattr(self, '_glotaran_model_attributes')

        errors = []

        for attr in attrs:
            for _, item in getattr(self, attr).items():
                item.validate_model(self, errors=errors)

        return errors

    def valid(self) -> bool:
        """valid checks the model for errors.

        Returns
        -------
        valid : bool
            False if at least one error in the model, else True.
        """
        return len(self.errors()) is 0

    def errors_parameter(self, parameter: ParameterGroup) -> List[str]:
        """errors_parameter returns a list of missing parameters.

        Parameters
        ----------
        parameter : ParameterGroup

        Returns
        -------
        errors : List[str]
        """
        attrs = getattr(self, '_glotaran_model_attributes')

        errors = []

        for attr in attrs:
            for _, item in getattr(self, attr).items():
                item.validate_parameter(self, parameter, errors=errors)

        return errors

    def valid_parameter(self, parameter):
        """valid checks the parameter for errors.

        Parameters
        ----------
        parameter : ParameterGroup

        Returns
        -------
        valid : bool
            False if at least one error in the parameter, else True.
        """
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
