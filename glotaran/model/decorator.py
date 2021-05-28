"""The model decorator."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List

from glotaran.model.attribute import model_attribute_typed
from glotaran.model.dataset_descriptor import DatasetDescriptor
from glotaran.model.megacomplex import Megacomplex
from glotaran.model.util import wrap_func_as_method
from glotaran.model.weight import Weight
from glotaran.plugin_system.model_registration import register_model

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Tuple
    from typing import Type
    from typing import Union

    import numpy as np
    import xarray as xr

    from glotaran.analysis.problem import Problem
    from glotaran.model.base_model import Model
    from glotaran.parameter import ParameterGroup

    MegacomplexMatrixFunction = Callable[
        [Type[object], Type[Model], Type[DatasetDescriptor], dict[str, int], Any],
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

    FinalizeFunction = Callable[[Problem, Dict[str, xr.Dataset]], None]
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
    attributes: dict[str, Any] = None,
    dataset_type: type[DatasetDescriptor] = DatasetDescriptor,
    default_megacomplex_type: str = None,
    megacomplex_types: dict[str, Megacomplex] | type[Megacomplex] = None,
    global_matrix: GlobalMatrixFunction = None,
    model_dimension: str = None,
    global_dimension: str = None,
    has_matrix_constraints_function: Callable[[type[Model]], bool] = None,
    constrain_matrix_function: ConstrainMatrixFunction = None,
    retrieve_clp_function: RetrieveClpFunction = None,
    has_additional_penalty_function: Callable[[type[Model]], bool] = None,
    additional_penalty_function: PenaltyFunction = None,
    finalize_data_function: FinalizeFunction = None,
    grouped: bool | Callable[[type[Model]], bool] = False,
    index_dependent: bool | Callable[[type[Model]], bool] = False,
) -> Callable[[type[Model]], type[Model]]:
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

        _set_constraints_functions(
            cls, has_matrix_constraints_function, constrain_matrix_function, retrieve_clp_function
        )

        _set_additional_penalty_functions(
            cls, has_additional_penalty_function, additional_penalty_function
        )

        _set_grouped_and_indexdependent(cls, grouped, index_dependent)

        _set_dimensions(cls, model_type, model_dimension, global_dimension)

        if global_matrix:
            g_mat = wrap_func_as_method(cls, name="global_matrix")(global_matrix)
            g_mat = staticmethod(g_mat)
            setattr(cls, "global_matrix", g_mat)
        else:
            setattr(cls, "global_matrix", None)

        if not hasattr(cls, "_glotaran_model_attributes"):
            setattr(cls, "_glotaran_model_attributes", {})
        else:
            setattr(
                cls,
                "_glotaran_model_attributes",
                getattr(cls, "_glotaran_model_attributes").copy(),
            )

        megacomplex_cls = _set_megacomplexes(
            cls, model_type, default_megacomplex_type, megacomplex_types
        )

        # We add the standard attributes here.
        if not issubclass(dataset_type, DatasetDescriptor):
            raise ValueError(
                f"Dataset descriptor of model {model_type} is not a subclass of DatasetDescriptor"
            )
        attributes["dataset"] = dataset_type
        attributes["megacomplex"] = megacomplex_cls
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


def _create_add_func(cls, name, item_type):
    @wrap_func_as_method(cls, name=f"add_{name}", annotations={"item": item_type})
    def add_item(self, item: item_type):
        f"""Adds an `{item_type.__name__}` object.

        Parameters
        ----------
        item :
            The `{item_type.__name__}` item.
        """

        if not isinstance(item, item_type) and (
            not hasattr(item_type, "_glotaran_model_attribute_typed")
            or not isinstance(item, tuple(item_type._glotaran_model_attribute_types.values()))
        ):
            raise TypeError
        getattr(self, f"_{name}").append(item)

    return add_item


def _create_get_func(cls, name, item_type):
    @wrap_func_as_method(cls, name=f"get_{name}", annotations={"return": item_type})
    def get_item(self, label: str) -> item_type:
        f"""
        Returns the `{item_type.__name__}` object with the given label.

        Parameters
        ----------
        label :
            The label of the `{item_type.__name__}` object.
        """
        return getattr(self, f"_{name}")[label]

    return get_item


def _create_set_func(cls, name, item_type):
    @wrap_func_as_method(cls, name=f"set_{name}", annotations={"item": item_type})
    def set_item(self, label: str, item: item_type):
        f"""
        Sets the `{item_type.__name__}` object with the given label with to the item.

        Parameters
        ----------
        label :
            The label of the `{item_type.__name__}` object.
        item :
            The `{item_type.__name__}` item.
        """

        if (
            not isinstance(item, item_type)
            and (
                not hasattr(item_type, "_glotaran_model_attribute_typed")
                or not isinstance(item, tuple(item_type._glotaran_model_attribute_types.values()))
            )
            and not isinstance(item, Megacomplex)
        ):
            raise TypeError
        getattr(self, f"_{name}")[label] = item

    return set_item


def _set_megacomplexes(cls, model_type, default_megacomplex_type, megacomplex_types):
    @model_attribute_typed({})
    class MetaMegacomplex:
        """This class holds all Megacomplex types defined by a model."""

    if not isinstance(megacomplex_types, dict):
        megacomplex_types = {model_type: megacomplex_types}
    for name, megacomplex_type in megacomplex_types.items():
        if not issubclass(megacomplex_type, Megacomplex):
            raise TypeError(
                f"Megacomplex type {name}(megacomplex_type) is not a subclass of Megacomplex"
            )
        MetaMegacomplex.add_type(name, megacomplex_type)

    if default_megacomplex_type is None:
        default_megacomplex_type = next(iter(megacomplex_types.keys()))
    setattr(MetaMegacomplex, "_glotaran_model_attribute_default_type", default_megacomplex_type)
    return MetaMegacomplex


def _create_property_for_attribute(cls, name, attribute_type):

    return_type = (
        Dict[str, attribute_type]
        if hasattr(attribute_type, "_glotaran_has_label")
        else List[attribute_type]
    )

    doc_type = "dictionary" if hasattr(attribute_type, "_glotaran_has_label") else "list"

    @property
    @wrap_func_as_method(cls, name=f"{name}", annotations={"return": return_type})
    def attribute(self) -> return_type:
        f"""A {doc_type} containing {type.__name__}"""
        return getattr(self, f"_{name}")

    return attribute


def _set_constraints_functions(
    cls, has_matrix_constraints_function, constrain_matrix_function, retrieve_clp_function
):
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


def _set_additional_penalty_functions(
    cls, has_additional_penalty_function, additional_penalty_function
):
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


def _set_grouped_and_indexdependent(cls, grouped, index_dependent):
    setattr(
        cls,
        "grouped",
        grouped if callable(grouped) else lambda model: grouped,
    )

    setattr(
        cls,
        "index_dependent",
        index_dependent if callable(index_dependent) else lambda model: index_dependent,
    )


def _set_dimensions(cls, model_type, model_dimension, global_dimension):
    if model_dimension is None:
        raise ValueError(f"Model dimension not specified for model {model_type}")
    setattr(cls, "model_dimension", model_dimension)

    if global_dimension is None:
        raise ValueError(f"Global dimension not specified for model {model_type}")
    setattr(cls, "global_dimension", global_dimension)
