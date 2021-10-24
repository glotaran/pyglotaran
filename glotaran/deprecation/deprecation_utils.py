"""Helper functions to give deprecation warnings."""

from __future__ import annotations

import os
import sys
from functools import wraps
from importlib import import_module
from importlib.metadata import distribution
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Hashable
from typing import Mapping
from typing import MutableMapping
from typing import TypeVar
from typing import cast
from warnings import warn

import numpy as np

DecoratedCallable = TypeVar(
    "DecoratedCallable", bound=Callable[..., Any]
)  # decorated function or class

if TYPE_CHECKING:
    from typing import NoReturn
    from typing import Sequence


class OverDueDeprecation(Exception):
    """Error thrown when a deprecation should have been removed.

    See Also
    --------
    deprecate
    warn_deprecated
    deprecate_module_attribute
    deprecate_submodule
    deprecate_dict_entry
    """


class GlotaranDeprectedApiError(Exception):
    """Exception raised when a deprecation has no replacement.

    See Also
    --------
    deprecate
    warn_deprecated
    deprecate_module_attribute
    deprecate_submodule
    deprecate_dict_entry
    """


class GlotaranApiDeprecationWarning(UserWarning):
    """Warning to give users about API changes.

    See Also
    --------
    deprecate
    warn_deprecated
    deprecate_module_attribute
    deprecate_submodule
    deprecate_dict_entry
    """


def glotaran_version() -> str:
    """Version of the distribution.

    This is basically the same as ``glotaran.__version__`` but independent from glotaran.
    This way all of the deprecation functionality can be used even in
    ``glotaran.__init__.py`` without moving the import below the definition of
    ``__version__`` or causeing a circular import issue.

    Returns
    -------
    str
        The version string.
    """
    return distribution("pyglotaran").version


def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse version string to tuple of three ints for comparison.

    Parameters
    ----------
    version_str : str
        Fully qualified version string of the form 'major.minor.patch'.

    Returns
    -------
    tuple[int, int, int]
        Version as tuple.

    Raises
    ------
    ValueError
        If ``version_str`` has less that three elements separated by ``.``.
    ValueError
        If ``version_str`` 's first three elements can not be casted to int.
    """
    error_message = (
        "version_str needs to be a fully qualified version consisting of "
        f"int parts (e.g. '0.0.1'), got {version_str!r}"
    )
    split_version = version_str.partition("-")[0].split(".")
    if len(split_version) < 3:
        raise ValueError(error_message)
    try:
        return tuple(
            map(int, (*split_version[:2], split_version[2].partition("rc")[0]))
        )  # type:ignore[return-value]
    except ValueError:
        raise ValueError(error_message)


def check_qualnames_in_tests(qual_names: Sequence[str], importable_indices: Sequence[int]):
    """Test that qualnames import path exists when running tests.

    All deprecations should be tested anyway in order to get the proper
    errors when a deprecation is overdue.
    This helperfunction also helps to ensure that at least the import
    paths (``qual_names``) of the old and new usage exist.

    Parameters
    ----------
    qual_names : Sequence[str]
        Sequence of fully qualified module attribute names,
        optionally with call arguments.
    importable_indices: Sequence[int]
        Indices of corresponding to ``qual_names`` indicating
        how to slice each ``qual_name`` split at ``.``, for the import
        and attribute checking.

    See Also
    --------
    warn_deprecated
    deprecate
    """
    # Since this is always true for tests run with pytest we ignore the branch coverage
    if "PYTEST_CURRENT_TEST" in os.environ:  # pragma: no branch
        for qual_name, slice_index in zip(qual_names, importable_indices):
            qual_name_parts = qual_name.partition("(")[0].partition("[")[0].split(".")
            module_name = ".".join(qual_name_parts[:-slice_index])
            object_name = qual_name_parts[-slice_index]
            module = __import__(module_name, fromlist=(object_name))
            assert hasattr(module, object_name)
            if slice_index != 1:
                item = getattr(module, object_name)
                hasattr(item, qual_name_parts[-slice_index + 1])


def check_overdue(deprecated_qual_name_usage: str, to_be_removed_in_version: str) -> None:
    """Check if a deprecation is overdue for removal.

    Parameters
    ----------
    deprecated_qual_name_usage : str
        Old usage with fully qualified name e.g.:
        ``'glotaran.read_model_from_yaml(model_yml_str)'``
    to_be_removed_in_version : str
        Version the support for this usage will be removed.

    Raises
    ------
    OverDueDeprecation
        If the current version is greater or equal to ``to_be_removed_in_version``.
    """
    if (
        parse_version(glotaran_version()) >= parse_version(to_be_removed_in_version)
        and "dev" not in glotaran_version()
    ):
        raise OverDueDeprecation(
            f"Support for {deprecated_qual_name_usage.partition('(')[0]!r} was "
            f"supposed to be dropped in version: {to_be_removed_in_version!r}.\n"
            f"Current version is: {glotaran_version()!r}"
        )


def raise_deprecation_error(
    *,
    deprecated_qual_name_usage: str,
    new_qual_name_usage: str,
    to_be_removed_in_version: str,
) -> NoReturn:
    """Raise :class:`GlotaranDeprectedApiError` error, with formatted message.

    This should only be used if there is no reasonable way to keep the deprecated
    usage functional!

    Parameters
    ----------
    deprecated_qual_name_usage : str
        Old usage with fully qualified name e.g.:
        ``'glotaran.read_model_from_yaml(model_yml_str)'``
    new_qual_name_usage : str
        New usage as fully qualified name e.g.:
        ``'glotaran.io.load_model(model_yml_str, format_name="yml_str")'``
    to_be_removed_in_version : str
        Version the support for this usage will be removed.

    Raises
    ------
    OverDueDeprecation
        If the current version is greater or equal to ``to_be_removed_in_version``.
    GlotaranDeprectedApiError
        If :class:`OverDueDeprecation` wasn't raised before.


    .. # noqa: DAR402 OverDueDeprecation
    .. # noqa: DAR401 GlotaranDeprectedApiError
    """
    check_overdue(deprecated_qual_name_usage, to_be_removed_in_version)
    message = (
        f"Usage of {deprecated_qual_name_usage!r} was deprecated, "
        f"use {new_qual_name_usage!r} instead.\n"
        "It wasn't possible to restore the original behavior of this usage "
        "(mostlikely due to an object hierarchy change)."
        "This usage change message won't be show as of version: "
        f"{to_be_removed_in_version!r}."
    )
    raise GlotaranDeprectedApiError(message)


def warn_deprecated(
    *,
    deprecated_qual_name_usage: str,
    new_qual_name_usage: str,
    to_be_removed_in_version: str,
    check_qual_names: tuple[bool, bool] = (True, True),
    stacklevel: int = 2,
    importable_indices: tuple[int, int] = (1, 1),
) -> None:
    """Raise deprecation warning with change information.

    The change information are old / new usage information and end of support version.

    Parameters
    ----------
    deprecated_qual_name_usage : str
        Old usage with fully qualified name e.g.:
        ``'glotaran.read_model_from_yaml(model_yml_str)'``
    new_qual_name_usage : str
        New usage as fully qualified name e.g.:
        ``'glotaran.io.load_model(model_yml_str, format_name="yml_str")'``
    to_be_removed_in_version : str
        Version the support for this usage will be removed.
    check_qual_names : tuple[bool, bool]
        Whether or not to check for the existence ``deprecated_qual_name_usage`` and
        ``deprecated_qual_name_usage``

        *   Set the first value to False to prevent infinite recursion error when changing
            a module attribute import.

        *   Set the second value to False if the new usage in in a different package or
            there is none.

    stacklevel: int
        Stack at which the warning should be shown as raise. Default: 2

    importable_indices : tuple[int, int]
        Indices from right for most nested item which is importable for
        ``deprecated_qual_name_usage`` and ``new_qual_name_usage``
        after splitting at ``.``. This is used when the old or new usage
        is a method or mapping access. E.g. let ``deprecated_qual_name_usage``
        be ``package.module.class.mapping["key"]``, then you would use
        ``importable_indices=(2, 1)``, this way func:`check_qualnames_in_tests`
        will import ``package.module.class`` and check if ``class`` has an attribute
        ``mapping``.

    Raises
    ------
    OverDueDeprecation
        If the current version is greater or equal to ``to_be_removed_in_version``.

    See Also
    --------
    deprecate
    deprecate_module_attribute
    deprecate_submodule
    check_qualnames_in_tests

    Examples
    --------
    This is the way the old ``read_parameters_from_yaml_file`` could deprecated and the usage of
    ``load_model`` being promoted instead.


    .. code-block:: python
        :caption: glotaran/deprecation/modules/glotaran_root.py

        def read_parameters_from_yaml_file(model_path: str):
            warn_deprecated(
                deprecated_qual_name_usage="glotaran.read_parameters_from_yaml_file(model_path)",
                new_qual_name_usage="glotaran.io.load_model.load_model(model_path)",
                to_be_removed_in_version="0.6.0",
            )
            return load_model(model_path)


    .. # noqa: DAR402
    """
    check_overdue(deprecated_qual_name_usage, to_be_removed_in_version)
    qual_names = (deprecated_qual_name_usage, new_qual_name_usage)
    selected_qual_names = [
        qual_name for qual_name, check in zip(qual_names, check_qual_names) if check
    ]
    selected_indices = importable_indices[: len(selected_qual_names)]
    check_qualnames_in_tests(qual_names=selected_qual_names, importable_indices=selected_indices)
    warn(
        GlotaranApiDeprecationWarning(
            f"Usage of {deprecated_qual_name_usage!r} was deprecated, "
            f"use {new_qual_name_usage!r} instead.\n"
            f"This usage will be an error in version: {to_be_removed_in_version!r}."
        ),
        stacklevel=stacklevel,
    )


def deprecate(
    *,
    deprecated_qual_name_usage: str,
    new_qual_name_usage: str,
    to_be_removed_in_version: str,
    has_glotaran_replacement: bool = True,
    importable_indices: tuple[int, int] = (1, 1),
) -> Callable[[DecoratedCallable], DecoratedCallable]:
    """Decorate a function, method or class to deprecate it.

    This raises deprecation warning with old / new usage information and
    end of support version.


    Parameters
    ----------
    deprecated_qual_name_usage : str
        Old usage with fully qualified name e.g.:
        ``'glotaran.read_model_from_yaml(model_yml_str)'``
    new_qual_name_usage : str
        New usage as fully qualified name e.g.:
        ``'glotaran.io.load_model(model_yml_str, format_name="yml_str")'``
    to_be_removed_in_version : str
        Version the support for this usage will be removed.
    has_glotaran_replacement : bool
        Whether or not this functionality has a replacement in core
        pyglotaran. This will be mapped to the second entry of ``check_qualnames``
        in :func:`warn_deprecated`.
    importable_indices : Sequence[int]
        Indices from right for most nested item which is importable for
        ``deprecated_qual_name_usage`` and ``new_qual_name_usage``
        after splitting at ``.``. This is used when the old or new usage
        is a method or mapping access. E.g. let ``deprecated_qual_name_usage``
        be ``package.module.class.mapping["key"]``, then you would use
        ``importable_indices=(2, 1)``, this way func:`check_qualnames_in_tests`
        will import ``package.module.class`` and check if ``class`` has an attribute
        ``mapping``. Default


    Returns
    -------
    DecoratedCallable
        Original function or class throwing a Deprecation warning when used.

    Raises
    ------
    OverDueDeprecation
        If the current version is greater or equal to ``to_be_removed_in_version``.

    See Also
    --------
    warn_deprecated
    deprecate_module_attribute
    deprecate_submodule
    check_qualnames_in_tests

    Examples
    --------
    This is the way the old ``read_parameters_from_yaml_file`` was deprecated and the usage of
    ``load_model`` was promoted instead.


    .. code-block:: python
        :caption: glotaran/deprecation/modules/glotaran_root.py

        @deprecate(
            deprecated_qualname_usage="glotaran.read_parameters_from_yaml_file(model_path)",
            new_qualname_usage="glotaran.io.load_model(model_path)",
            to_be_removed_in_version="0.6.0",
        )
        def read_parameters_from_yaml_file(model_path: str):
            return load_model(model_path)


    .. # noqa: DAR402
    """

    def inject_warn_into_call(deprecated_object: DecoratedCallable) -> DecoratedCallable:
        """Wrap warning into function call.

        Used on deprecated_object.__new__ if it's a class or else on deprecated_object.
        """

        @wraps(deprecated_object)
        def inner_wrapper(*args: Any, **kwargs: Any) -> DecoratedCallable:
            """Wrap running the function and warning."""
            warn_deprecated(
                deprecated_qual_name_usage=deprecated_qual_name_usage,
                new_qual_name_usage=new_qual_name_usage,
                to_be_removed_in_version=to_be_removed_in_version,
                stacklevel=3,
                check_qual_names=(True, has_glotaran_replacement),
                importable_indices=importable_indices,
            )
            return deprecated_object(*args, **kwargs)

        return cast(DecoratedCallable, inner_wrapper)

    def outer_wrapper(deprecated_object: DecoratedCallable) -> DecoratedCallable:
        """Wrap deprecated_object of all callable kinds."""
        if not isinstance(deprecated_object, type):
            return cast(DecoratedCallable, inject_warn_into_call(deprecated_object))

        setattr(
            deprecated_object,
            "__new__",
            inject_warn_into_call(deprecated_object.__new__),  # type: ignore[arg-type]
        )
        return deprecated_object  # type: ignore[return-value]

    return cast(Callable[[DecoratedCallable], DecoratedCallable], outer_wrapper)


def deprecate_dict_entry(
    *,
    dict_to_check: MutableMapping[Hashable, Any],
    deprecated_usage: str,
    new_usage: str,
    to_be_removed_in_version: str,
    swap_keys: tuple[Hashable, Hashable] | None = None,
    replace_rules: tuple[Mapping[Hashable, Any], Mapping[Hashable, Any]] | None = None,
    stacklevel: int = 3,
) -> None:
    """Replace dict entry inplace and warn about usage change, if present in the dict.

    Parameters
    ----------
    dict_to_check : MutableMapping[Hashable, Any]
        Dict which should be checked.
    deprecated_usage : str
        Old usage to inform user (only used in warning).
    new_usage : str
        New usage to inform user (only used in warning).
    to_be_removed_in_version : str
        Version the support for this usage will be removed.
    swap_keys : tuple[Hashable, Hashable]
        (old_key, new_key),
        ``dict_to_check[new_key]`` will be assigned the value ``dict_to_check[old_key]``
        and ``old_key`` will be removed from the dict.
        by default None
    replace_rules : Mapping[Hashable, tuple[Any, Any]]
        ({old_key: old_value}, {new_key: new_value}),
        If ``dict_to_check[old_key]`` has the value ``old_value``,
        ``dict_to_check[new_key]`` it will be set to ``new_value``.
        ``old_key`` will be removed from the dict if ``old_key`` and ``new_key`` aren't equal.
        by default None
    stacklevel : int
        Stack at which the warning should be shown as raise. , by default 3


    Raises
    ------
    ValueError
        If both ``swap_keys`` and ``replace_rules`` are None (default) or not None.
    OverDueDeprecation
        If the current version is greater or equal to ``to_be_removed_in_version``.

    See Also
    --------
    warn_deprecated

    Notes
    -----
    To prevent confusion exactly one of ``replace_rules`` and ``swap_keys``
    needs to be passed.

    Examples
    --------
    For readability sake the warnings won't be shown in the examples.

    Swapping key names:

    >>> dict_to_check = {"foo": 123}
    >>> deprecate_dict_entry(
            dict_to_check=dict_to_check,
            deprecated_usage="foo",
            new_usage="bar",
            to_be_removed_in_version="0.6.0",
            swap_keys=("foo", "bar")
        )
    >>> dict_to_check
    {"bar": 123}

    Changing values:

    >>> dict_to_check = {"foo": 123}
    >>> deprecate_dict_entry(
            dict_to_check=dict_to_check,
            deprecated_usage="foo: 123",
            new_usage="foo: 123.0",
            to_be_removed_in_version="0.6.0",
            replace_rules=({"foo": 123}, {"foo": 123.0})
        )
    >>> dict_to_check
    {"foo": 123.0}

    Swapping key names AND changing values:

    >>> dict_to_check = {"type": "kinetic-spectrum"}
    >>> deprecate_dict_entry(
            dict_to_check=dict_to_check,
            deprecated_usage="type: kinectic-spectrum",
            new_usage="default_megacomplex: decay",
            to_be_removed_in_version="0.6.0",
            replace_rules=({"type": "kinetic-spectrum"}, {"default_megacomplex": "decay"})
        )
    >>> dict_to_check
    {"default_megacomplex": "decay"}


    .. # noqa: DAR402
    """
    dict_changed = False

    if not np.logical_xor(swap_keys is None, replace_rules is None):
        raise ValueError(
            "Exactly one of the parameters `swap_keys` or `replace_rules` needs to be provided."
        )
    if swap_keys is not None and swap_keys[0] in dict_to_check:
        dict_changed = True
        dict_to_check[swap_keys[1]] = dict_to_check[swap_keys[0]]
        del dict_to_check[swap_keys[0]]
    if replace_rules is not None:
        old_key, old_value = next(iter(replace_rules[0].items()))
        new_key, new_value = next(iter(replace_rules[1].items()))
        if old_key in dict_to_check and dict_to_check[old_key] == old_value:
            dict_changed = True
            dict_to_check[new_key] = new_value
            if new_key != old_key:
                del dict_to_check[old_key]

    if dict_changed:
        warn_deprecated(
            deprecated_qual_name_usage=deprecated_usage,
            new_qual_name_usage=new_usage,
            to_be_removed_in_version=to_be_removed_in_version,
            stacklevel=stacklevel,
            check_qual_names=(False, False),
        )


def module_attribute(module_qual_name: str, attribute_name: str) -> Any:
    """Import and return the attribute (e.g. function or class) of a module.

    This is basically the same as ``from module_name import attribute_name as return_value``
    where this function returns ``return_value``.

    Parameters
    ----------
    module_qual_name : str
        Fully qualified name for a module e.g. ``glotaran.model.base_model``
    attribute_name : str
        Name of the attribute e.g. ``Model``

    Returns
    -------
    Any
        Attribute of the module, e.g. a function or class.
    """
    module = import_module(module_qual_name)
    return getattr(module, attribute_name)


def deprecate_module_attribute(
    *,
    deprecated_qual_name: str,
    new_qual_name: str,
    to_be_removed_in_version: str,
) -> Any:
    """Import and return and anttribute from the new location.

    This needs to be wrapped in the definition of a module wide
    ``__getattr__`` function so it won't throw warnings all the time
    (see example).

    Parameters
    ----------
    deprecated_qual_name : str
        Fully qualified name of the deprecated attribute e.g.:
        ``glotaran.ParameterGroup``
    new_qual_name : str
        Fully qualified name of the new attribute e.g.:
        ``glotaran.parameter.ParameterGroup``
    to_be_removed_in_version : str
        Version the support for this usage will be removed.

    Returns
    -------
    Any
        Module attribute from its new location.

    Raises
    ------
    OverDueDeprecation
        If the current version is greater or equal to ``to_be_removed_in_version``.

    See Also
    --------
    deprecate
    warn_deprecated
    deprecate_submodule

    Examples
    --------
    When deprecating the usage of ``ParameterGroup`` the root of ``glotaran``
    and promoting to import it from ``glotaran.parameter`` the following code
    was added to the root ``__init__.py``.

    .. code-block:: python
        :caption: glotaran/__init__.py


        def __getattr__(attribute_name: str):
            from glotaran.deprecation import deprecate_module_attribute

            if attribute_name == "ParameterGroup":
                return deprecate_module_attribute(
                    deprecated_qual_name="glotaran.ParameterGroup",
                    new_qual_name="glotaran.parameter.ParameterGroup",
                    to_be_removed_in_version="0.6.0",
                )

            raise AttributeError(f"module {__name__} has no attribute {attribute_name}")


    .. # noqa: DAR402
    """
    module_name = ".".join(new_qual_name.split(".")[:-1])
    attribute_name = new_qual_name.split(".")[-1]

    warn_deprecated(
        deprecated_qual_name_usage=deprecated_qual_name,
        new_qual_name_usage=new_qual_name,
        to_be_removed_in_version=to_be_removed_in_version,
        check_qual_names=(False, True),
        stacklevel=4,
        importable_indices=(1, 1),
    )
    return module_attribute(module_name, attribute_name)


def deprecate_submodule(
    *,
    deprecated_module_name: str,
    new_module_name: str,
    to_be_removed_in_version: str,
) -> ModuleType:
    r"""Create a module at runtime which retrieves attributes from new module.

    When moving a module, create a variable with the modules name in the
    parent packages ``__init__.py``, so imports will be redirected to the
    new module location and a deprecation warning will be given, to help
    the user adjust the outdated code.
    Each time an attribute is retrieved there will be a deprecation warning.

    Parameters
    ----------
    deprecated_module_name : str
        Fully qualified name of the deprecated module e.g.:
        ``'glotaran.analysis.result'``
    new_module_name : str
        Fully qualified name of the new module e.g.:
        ``'glotaran.project.result'``
    to_be_removed_in_version : str
        Version the support for this usage will be removed.

    Returns
    -------
    ModuleType
        Module containing

    Raises
    ------
    OverDueDeprecation
        If the current version is greater or equal to ``to_be_removed_in_version``.

    See Also
    --------
    deprecate
    deprecate_module_attribute

    Examples
    --------
    When moving the module ``result`` from ``glotaran.analysis.result`` to
    ``glotaran.project.result`` the following code was added to the old parent
    packages (``glotaran.analysis``) ``__init__.py``.

    .. code-block:: python
        :caption: glotaran/analysis/__init__.py

        from glotaran.deprecation.deprecation_utils import deprecate_submodule

        result = deprecate_submodule(
            deprecated_module_name="glotaran.analysis.result",
            new_module_name="glotaran.project.result",
            to_be_removed_in_version="0.6.0",
        )


    .. # noqa: DAR402
    """
    new_module = import_module(new_module_name)
    deprecated_module = ModuleType(
        deprecated_module_name,
        f"Deprecated use {new_module_name!r} instead.\n\n{new_module.__doc__}",
    )

    def warn_getattr(attribute_name: str):

        if attribute_name == "__file__":
            return new_module.__file__

        elif attribute_name in dir(new_module):
            return deprecate_module_attribute(
                deprecated_qual_name=f"{deprecated_module_name}.{attribute_name}",
                new_qual_name=f"{new_module_name}.{attribute_name}",
                to_be_removed_in_version=to_be_removed_in_version,
            )

        raise AttributeError(f"module {deprecated_module_name} has no attribute {attribute_name}")

    setattr(deprecated_module, "__getattr__", warn_getattr)
    setattr(deprecated_module, "__package__", deprecated_module_name.split(".")[:-1])
    setattr(deprecated_module, "__dir__", new_module.__dir__)

    sys.modules[deprecated_module_name] = deprecated_module
    return deprecated_module
