"""The model decorator."""

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np
import xarray as xr

import glotaran  # TODO: refactor to postponed type annotation
from glotaran.parameter import ParameterGroup
from glotaran.parse.register import register_model

from .base_model import Model
from .dataset_descriptor import DatasetDescriptor
from .util import wrap_func_as_method
from .weight import Weight

MatrixFunction = Callable[[Type[DatasetDescriptor], xr.Dataset], Tuple[List[str], np.ndarray]]
"""A `MatrixFunction` calculates the matrix for a model."""

IndexDependentMatrixFunction = Callable[
    [Type[DatasetDescriptor], xr.Dataset, Any],
    Tuple[List[str], np.ndarray],
]
"""A `MatrixFunction` calculates the matrix for a model."""

GlobalMatrixFunction = Callable[
    [Type[DatasetDescriptor], np.ndarray], Tuple[List[str], np.ndarray]
]
"""A `GlobalMatrixFunction` calculates the global matrix for a model."""

ConstrainMatrixFunction = Callable[
    [Type[Model], ParameterGroup, List[str], np.ndarray, float],
    Tuple[List[str], np.ndarray],
]
"""A `ConstrainMatrixFunction` applies constraints on a matrix."""

RetrieveClpFunction = Callable[
    [
        Type[Model],
        ParameterGroup,
        Dict[str, Union[List[str], List[List[str]]]],
        Dict[str, Union[List[str], List[List[str]]]],
        Dict[str, List[np.ndarray]],
        Dict[str, xr.Dataset],
    ],
    Dict[str, List[np.ndarray]],
]
"""A `RetrieveClpFunction` retrieves the full set of clp from a reduced set."""

FinalizeFunction = Callable[
    [TypeVar("glotaran.analysis.problem.Problem"), Dict[str, xr.Dataset]], None
]
"""A `FinalizeFunction` gets called after optimization."""

PenaltyFunction = Callable[
    [
        Type[Model],
        ParameterGroup,
        Dict[str, Union[List[str], List[List[str]]]],
        Dict[str, List[np.ndarray]],
        Dict[str, Union[np.ndarray, List[np.ndarray]]],
        Dict[str, xr.Dataset],
        float,
    ],
    np.ndarray,
]
"""A `PenaltyFunction` calculates additional penalties for the optimization."""


def model(
    model_type: str,
    attributes: Dict[str, Any] = None,
    dataset_type: Type[DatasetDescriptor] = DatasetDescriptor,
    megacomplex_type: Any = None,
    matrix: Union[MatrixFunction, IndexDependentMatrixFunction] = None,
    global_matrix: GlobalMatrixFunction = None,
    model_dimension: str = None,
    global_dimension: str = None,
    has_matrix_constraints_function: Callable[[Type[Model]], bool] = None,
    constrain_matrix_function: ConstrainMatrixFunction = None,
    retrieve_clp_function: RetrieveClpFunction = None,
    has_additional_penalty_function: Callable[[Type[Model]], bool] = None,
    additional_penalty_function: PenaltyFunction = None,
    finalize_data_function: FinalizeFunction = None,
    grouped: Union[bool, Callable[[Type[Model]], bool]] = False,
    index_dependent: Union[bool, Callable[[Type[Model]], bool]] = False,
) -> Callable:
    """The `@model` decorator is intended to be used on subclasses of :class:`glotaran.model.Model`.
    It creates properties for the given attributes as well as functions to add access them. Also it
    adds the functions (e.g. for `matrix`) to the model ensures they are added wrapped in a correct
    way.

    Parameters
    ----------
    model_type : str
        Human readable string used by the parser to identify the correct model.
    attributes : Dict[str, Any], optional
        A dictionary of attribute names and types. All types must be decorated with the
        :func:`glotaran.model.model_attribute` decorator, by default None.
    dataset_type : Type[DatasetDescriptor], optional
        A subclass of :class:`DatasetDescriptor`, by default DatasetDescriptor
    megacomplex_type : Any, optional
        A class for the model megacomplexes. The class must be decorated with the
        :func:`glotaran.model.model_attribute` decorator, by default None
    matrix : Union[MatrixFunction, IndexDependentMatrixFunction], optional
        A function to calculate the matrix for the model, by default None
    global_matrix : GlobalMatrixFunction, optional
        A function to calculate the global matrix for the model, by default None
    model_dimension : str, optional
        The name of model matrix row dimension, by default None
    global_dimension : str, optional
        The name of model global matrix row dimension, by default None
    has_matrix_constraints_function : Callable[[Type[Model]], bool], optional
        True if the model as a constrain_matrix_function set, by default None
    constrain_matrix_function : ConstrainMatrixFunction, optional
        A function to constrain the global matrix for the model, by default None
    retrieve_clp_function : RetrieveClpFunction, optional
        A function to retrieve the full clp from the reduced, by default None
    has_additional_penalty_function : Callable[[Type[Model]], bool], optional
        True if model has a additional_penalty_function set, by default None
    additional_penalty_function : PenaltyFunction, optional
        A function to calculate additional penalties when optimizing the model, by default None
    finalize_data_function : FinalizeFunction, optional
        A function to finalize data after optimization, by default None
    grouped : Union[bool, Callable[[Type[Model]], bool]], optional
        True if model described a grouped problem, by default False
    index_dependent : Union[bool, Callable[[Type[Model]], bool]], optional
        True if model described a index dependent problem, by default False

    Returns
    -------
    Callable
        Returns a decorated model function

    Raises
    ------
    ValueError
        If model implements meth:`has_matrix_constraints_function` but not
        meth:`constrain_matrix_function` and meth:`retrieve_clp_function`
    ValueError
        If model implements meth:`has_additional_penalty_function` but not
        meth:`additional_penalty_function`
    """

    def decorator(cls):

        setattr(cls, "_model_type", model_type)
        setattr(cls, "finalize_data", finalize_data_function)

        if has_matrix_constraints_function:
            if not constrain_matrix_function:
                raise ValueError(
                    "Model implements `has_matrix_constraints_function` "
                    "but not `constrain_matrix_function`"
                )
            if not retrieve_clp_function:
                raise ValueError(
                    "Model implements `has_matrix_constraints_function` "
                    "but not `retrieve_clp_function`"
                )
            has_c_mat = wrap_func_as_method(cls, name="has_matrix_constraints_function")(
                has_matrix_constraints_function
            )
            c_mat = wrap_func_as_method(cls, name="constrain_matrix_function")(
                constrain_matrix_function
            )
            r_clp = wrap_func_as_method(cls, name="retrieve_clp_function")(retrieve_clp_function)
            setattr(cls, "has_matrix_constraints_function", has_c_mat)
            setattr(cls, "constrain_matrix_function", c_mat)
            setattr(cls, "retrieve_clp_function", r_clp)
        else:
            setattr(cls, "has_matrix_constraints_function", None)
            setattr(cls, "constrain_matrix_function", None)
            setattr(cls, "retrieve_clp_function", None)

        if has_additional_penalty_function:
            if not additional_penalty_function:
                raise ValueError(
                    "Model implements `has_additional_penalty_function`"
                    "but not `additional_penalty_function`"
                )
            has_pen = wrap_func_as_method(cls, name="has_additional_penalty_function")(
                has_additional_penalty_function
            )
            pen = wrap_func_as_method(cls, name="additional_penalty_function")(
                additional_penalty_function
            )
            setattr(cls, "additional_penalty_function", pen)
            setattr(cls, "has_additional_penalty_function", has_pen)
        else:
            setattr(cls, "has_additional_penalty_function", None)
            setattr(cls, "additional_penalty_function", None)

        if not callable(grouped):

            def group_fun(model):
                return grouped

        else:
            group_fun = grouped
        setattr(cls, "grouped", group_fun)

        if not callable(index_dependent):

            def index_dep_fun(model):
                return index_dependent

        else:
            index_dep_fun = index_dependent
        setattr(cls, "index_dependent", index_dep_fun)

        mat = wrap_func_as_method(cls, name="matrix")(matrix)
        mat = staticmethod(mat)
        setattr(cls, "matrix", mat)

        if model_dimension is None:
            raise ValueError(f"Model dimension not specified for model {model_type}")
        setattr(cls, "model_dimension", model_dimension)

        if global_matrix:
            g_mat = wrap_func_as_method(cls, name="global_matrix")(global_matrix)
            g_mat = staticmethod(g_mat)
            setattr(cls, "global_matrix", g_mat)
        else:
            setattr(cls, "global_matrix", None)

        if global_dimension is None:
            raise ValueError(f"Global dimension not specified for model {model_type}")
        setattr(cls, "global_dimension", global_dimension)

        if not hasattr(cls, "_glotaran_model_attributes"):
            setattr(cls, "_glotaran_model_attributes", {})
        else:
            setattr(
                cls,
                "_glotaran_model_attributes",
                getattr(cls, "_glotaran_model_attributes").copy(),
            )

        # We add the standard attributes here.
        attributes["dataset"] = dataset_type
        attributes["megacomplex"] = megacomplex_type
        attributes["weights"] = Weight

        # Set annotations and methods for attributes
        for attr_name, attr_type in attributes.items():

            # store for internal lookups
            getattr(cls, "_glotaran_model_attributes")[attr_name] = None

            # create and attach the property to class
            attr_prop = _create_property_for_attribute(cls, attr_name, attr_type)
            setattr(cls, attr_name, attr_prop)

            # properties with labels are implemented as dicts, whereas properties
            # without as arrays. Thus the need different setters.
            if getattr(attr_type, "_glotaran_has_label"):
                get_item = _create_get_func(cls, attr_name, attr_type)
                setattr(cls, get_item.__name__, get_item)
                set_item = _create_set_func(cls, attr_name, attr_type)
                setattr(cls, set_item.__name__, set_item)

            else:
                add_item = _create_add_func(cls, attr_name, attr_type)
                setattr(cls, add_item.__name__, add_item)

        init = _create_init_func(cls, attributes)
        setattr(cls, "__init__", init)

        register_model(model_type, cls)

        return cls

    return decorator


def _create_init_func(cls, attributes):
    @wrap_func_as_method(cls)
    def __init__(self):
        for attr_name, attr_item in attributes.items():
            if getattr(attr_item, "_glotaran_has_label"):
                setattr(self, f"_{attr_name}", {})
            else:
                setattr(self, f"_{attr_name}", [])
        super(cls, self).__init__()

    return __init__


def _create_add_func(cls, name, type):
    @wrap_func_as_method(cls, name=f"add_{name}")
    def add_item(self, item: type):
        f"""Adds an `{type.__name__}` object.

        Parameters
        ----------
        item :
            The `{type.__name__}` item.
        """

        if not isinstance(item, type) and (
            not hasattr(type, "_glotaran_model_attribute_typed")
            or not isinstance(item, tuple(type._glotaran_model_attribute_types.values()))
        ):
            raise TypeError
        getattr(self, f"_{name}").append(item)

    return add_item


def _create_get_func(cls, name, type):
    @wrap_func_as_method(cls, name=f"get_{name}")
    def get_item(self, label: str) -> type:
        f"""
        Returns the `{type.__name__}` object with the given label.

        Parameters
        ----------
        label :
            The label of the `{type.__name__}` object.
        """
        return getattr(self, f"_{name}")[label]

    return get_item


def _create_set_func(cls, name, type):
    @wrap_func_as_method(cls, name=f"set_{name}")
    def set_item(self, label: str, item: type):
        f"""
        Sets the `{type.__name__}` object with the given label with to the item.

        Parameters
        ----------
        label :
            The label of the `{type.__name__}` object.
        item :
            The `{type.__name__}` item.
        """

        if not isinstance(item, type) and (
            not hasattr(type, "_glotaran_model_attribute_typed")
            or not isinstance(item, tuple(type._glotaran_model_attribute_types.values()))
        ):
            raise TypeError
        getattr(self, f"_{name}")[label] = item

    return set_item


def _create_property_for_attribute(cls, name, type):

    return_type = Dict[str, type] if hasattr(type, "_glotaran_has_label") else List[type]

    doc_type = "dictionary" if hasattr(type, "_glotaran_has_label") else "list"

    @property
    @wrap_func_as_method(cls, name=f"{name}")
    def attribute(self) -> return_type:
        f"""A {doc_type} containing {type.__name__}"""
        return getattr(self, f"_{name}")

    return attribute
