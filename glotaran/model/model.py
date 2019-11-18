"""A base class for global analysis models."""


import copy
import inspect
import typing
import numpy as np
import xarray as xr

from glotaran.analysis.result import Result
from glotaran.analysis.scheme import Scheme
from glotaran.analysis.simulation import simulate
from glotaran.analysis.optimize import optimize
from glotaran.parameter import ParameterGroup


class Model:
    """A base class for global analysis models."""

    @classmethod
    def from_dict(cls, model_dict_ref: typing.Dict) -> typing.Type['Model']:
        """Creates a model from a dictionary.

        Parameters
        ----------
        model_dict :
            Dictionary containing the model.
        """

        model = cls()

        model_dict = copy.deepcopy(model_dict_ref)

        # iterate over items
        for name, attribute in list(model_dict.items()):

            # we determine if we the item is known by the model by looking for
            # a setter with same name.

            if hasattr(model, f'set_{name}'):

                # get the set function
                set = getattr(model, f'set_{name}')

                # we retrieve the actual class from the signature
                for label, item in attribute.items():
                    item_cls = set.__func__.__annotations__['item']
                    is_typed = hasattr(item_cls, "_glotaran_model_attribute_typed")
                    if isinstance(item, dict):
                        if is_typed:
                            if 'type' not in item:
                                raise Exception(f"Missing type for attribute '{name}'")
                            item_type = item['type']

                            if item_type not in item_cls._glotaran_model_attribute_types:
                                raise Exception(f"Unknown type '{item_type}' "
                                                f"for attribute '{name}'")
                            item_cls = \
                                item_cls._glotaran_model_attribute_types[item_type]
                        item['label'] = label
                        set(label, item_cls.from_dict(item))
                    elif isinstance(item, list):
                        if is_typed:
                            if len(item) < 2 and len(item) != 1:
                                raise Exception(f"Missing type for attribute '{name}'")
                            item_type = item[1] if len(item) != 1 and \
                                hasattr(item_cls, 'label') else item[0]

                            if item_type not in item_cls._glotaran_model_attribute_types:
                                raise Exception(f"Unknown type '{item_type}' "
                                                f"for attribute '{name}'")
                            item_cls = \
                                item_cls._glotaran_model_attribute_types[item_type]
                        item = [label] + item
                        set(label, item_cls.from_list(item))
                del model_dict[name]

            elif hasattr(model, f'add_{name}'):

                # get the set function
                add = getattr(model, f'add_{name}')

                # we retrieve the actual class from the signature
                for item in attribute:
                    item_cls = add.__func__.__annotations__['item']
                    is_typed = hasattr(item_cls, "_glotaran_model_attribute_typed")
                    if isinstance(item, dict):
                        if is_typed:
                            if 'type' not in item:
                                raise Exception(f"Missing type for attribute '{name}'")
                            item_type = item['type']

                            if item_type not in item_cls._glotaran_model_attribute_types:
                                raise Exception(f"Unknown type '{item_type}' "
                                                f"for attribute '{name}'")
                            item_cls = \
                                item_cls._glotaran_model_attribute_types[item_type]
                        add(item_cls.from_dict(item))
                    elif isinstance(item, list):
                        if is_typed:
                            if len(item) < 2 and len(item) != 1:
                                raise Exception(f"Missing type for attribute '{name}'")
                            item_type = item[1] if len(item) != 1 and \
                                hasattr(item_cls, 'label') else item[0]

                            if item_type not in item_cls._glotaran_model_attribute_types:
                                raise Exception(f"Unknown type '{item_type}' "
                                                f"for attribute '{name}'")
                            item_cls = \
                                item_cls._glotaran_model_attribute_types[item_type]
                        add(item_cls.from_list(item))
                del model_dict[name]

        return model

    @property
    def index_depended_matrix(self):
        return len(inspect.signature(self.matrix).parameters) == 3

    @property
    def model_type(self) -> str:
        """The type of the model as human readable string."""
        return self._model_type

    def simulate(self,
                 dataset: str,
                 parameter: ParameterGroup,
                 axes: typing.Dict[str, np.ndarray] = None,
                 clp: typing.Union[np.ndarray, xr.DataArray] = None,
                 noise: bool = False,
                 noise_std_dev: float = 1.0,
                 noise_seed: int = None,
                 ) -> xr.Dataset:
        """Simulates the model.

        Parameters
        ----------
        dataset :
            Label of the dataset to simulate.
        parameter :
            The parameters for the simulation.
        axes :
            A dictory with axes for simulation.
        clp :
            Conditionaly linear parameter. Will be used instead of `model.global_matrix` if given.
        noise :
            If `True` noise is added to the simulated data.
        noise_std_dev :
            The standart deviation of the noise.
        noise_seed :
            Seed for the noise.
        """
        return simulate(self, dataset, parameter, axes=axes, clp=clp, noise=noise,
                        noise_std_dev=noise_std_dev, noise_seed=noise_seed)

    def optimize(self,
                 parameter: ParameterGroup,
                 data: typing.Dict[str, typing.Union[xr.Dataset, xr.DataArray]],
                 nnls: bool = False,
                 verbose: bool = True,
                 max_nfev: int = None,
                 group_tolerance: int = 0,
                 client=None,
                 ) -> Result:
        """Optimizes the parameter for this model.

        Parameters
        ----------
        data :
            A dictonary containing all datasets with their labels as keys.
        parameter : glotaran.model.ParameterGroup
            The initial parameter.
        nnls :
            If `True` non-linear least squaes optimizing is used instead of variable projection.
        verbose :
            If `True` feedback is printed at every iteration.
        max_nfev :
            Maximum number of function evaluations. `None` for unlimited.
        group_tolerance :
            The tolerance for grouping datasets along the global dimension.
        """
        scheme = Scheme(model=self, parameter=parameter, data=data,
                        nnls=nnls, group_tolerance=group_tolerance, nfev=max_nfev)
        result = optimize(scheme, verbose=verbose, client=client)
        return result

    def result_from_parameter(self,
                              parameter: ParameterGroup,
                              data: typing.Dict[str, typing.Union[xr.DataArray, xr.Dataset]],
                              nnls: bool = False, group_atol: float = 0.0
                              ) -> Result:
        """Loads a result from parameters without optimization.

        Parameters
        ----------
        data :
            A dictonary containing all datasets with their labels as keys.
        parameter : glotaran.model.ParameterGroup
            The parameter.
        nnls :
            If `True` non-linear least squaes optimizing is used instead of variable projection.
        group_atol :
            The tolerance for grouping datasets along the global axes.

        """
        return Result.from_parameter(self, data, parameter, nnls, group_atol)

    def problem_list(self, parameter: ParameterGroup = None) -> typing.List[str]:
        """
        Returns a list with all problems in the model and missing parameters if specified.

        Parameters
        ----------

        parameter :
            The parameter to validate.
        """
        problems = []

        attrs = getattr(self, '_glotaran_model_attributes')
        for attr in attrs:
            attr = getattr(self, attr)
            if isinstance(attr, list):
                for item in attr:
                    problems += item.validate(self, parameter=parameter)
            else:
                for _, item in attr.items():
                    problems += item.validate(self, parameter=parameter)

        return problems

    def validate(self, parameter: ParameterGroup = None) -> str:
        """
        Returns a string listing all problems in the model and missing parameters if specified.

        Parameters
        ----------

        parameter :
            The parameter to validate.
        """
        result = ""

        problems = self.problem_list(parameter)
        if problems:
            result = f"Your model has {len(problems)} problems:\n"
            for p in problems:
                result += f"\n * {p}"
        else:
            result = "Your model is valid."
        return result

    def valid(self, parameter: ParameterGroup = None) -> bool:
        """Returns `True` if the number problems in the model is 0, else `False`

        Parameters
        ----------

        parameter :
            The parameter to validate.
        """
        return len(self.problem_list(parameter)) == 0

    def markdown(self, parameter: ParameterGroup = None, initial: ParameterGroup = None) -> str:
        """Formats the model as Markdown string.

        Parameters will be included if specified.

        Parameters
        ----------
        parameter :
            Parameter to include.
        initial :
            Initial values for the parameters.
        """
        attrs = getattr(self, '_glotaran_model_attributes')
        string = "# Model\n\n"
        string += f"_Type_: {self.model_type}\n\n"

        for attr in attrs:
            items = getattr(self, attr)
            if not items:
                continue

            string += f"## {attr.replace('_', ' ').title()}\n"
            string += "\n"

            if isinstance(items, dict):
                items = items.values()
            for item in items:
                item_str = item.mprint(parameter=parameter, initial=initial).split('\n')
                string += f'* {item_str[0]}\n'
                for s in item_str[1:]:
                    string += f"  {s}\n"
            string += "\n"
        return string

    def __str__(self):
        return self.markdown()
