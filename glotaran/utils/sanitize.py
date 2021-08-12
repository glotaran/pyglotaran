"""Glotaran module with utilities for sanitation of parsed content."""
from __future__ import annotations

from typing import Any

from glotaran.deprecation import warn_deprecated
from glotaran.utils.regex import RegexPattern as rp


def sanitize_list_with_broken_tuples(mangled_list: list[str | float]) -> list[str]:
    """Sanitize a list with 'broken' tuples.

    A list of broken tuples as returned by yaml when parsing tuples.
    e.g parsing the list of tuples [(3,100), (4,200)] results in
    a list of str ['(3', '100)', '(4', '200)'] which can be restored to
    a list with the tuples restored as strings ['(3, 100)', '(4, 200)']

    Parameters
    ----------
    mangled_list : List[Union[str,float]]
        A list with strings representing tuples broken up by round brackets.

    Returns
    -------
    List[str]
        A list containing the restores tuples (in string form) which can be
        converted back to numbered tuples using `list_string_to_tuple`
    """
    sanitized_string = str(mangled_list).replace("'", "")
    return list(rp.elements_in_string_of_list.findall(sanitized_string))


def sanitize_dict_keys(d: dict) -> dict:
    """Sanitize the stringified tuple dict keys in a yaml parsed dict.

    Keys representing a tuple, e.g. '(s1, s2)' are converted to a tuple of strings
        e.g. ('s1', 's2')

    Parameters
    ----------
    d : dict
        A dict containing tuple-like string keys

    Returns
    -------
    dict
        A dict with tuple-like string keys converted to tuple keys
    """
    if not isinstance(d, (dict, list)):
        return {}
    d_new = {}
    for k, v in d.items() if isinstance(d, dict) else enumerate(d):
        if isinstance(d, dict) and isinstance(k, str) and rp.tuple_word.match(k):
            k_new = tuple(map(str, rp.word.findall(k)))
            d_new.update({k_new: v})
        elif isinstance(d, (dict, list)):
            new_v = sanitize_dict_keys(v)
            if new_v:
                d[k] = new_v
    return d_new


def sanitize_dict_values(d: dict[str, Any] | list[Any]):
    """Sanitizes a dict with broken tuples inside modifying it in-place.

    Broken tuples are tuples that are turned into strings by the yaml parser.
    This functions calls `sanitize_list_with_broken_tuples` to glue the broken strings together
    and then calls list_to_tuple to turn the list with tuple strings back to number tuples.

    Parameters
    ----------
    d : dict
        A (complex) dict containing (possibly nested) values of broken tuple strings.
    """
    if not isinstance(d, (dict, list)):
        return
    for k, v in d.items() if isinstance(d, dict) else enumerate(d):  # type: ignore[attr-defined]
        if isinstance(v, list):
            leaf = all(isinstance(el, (str, tuple, float)) for el in v)
            if leaf:
                if "(" in str(v):
                    d[k] = list_string_to_tuple(sanitize_list_with_broken_tuples(v))
            else:
                sanitize_dict_values(v)
        if isinstance(v, dict):
            sanitize_dict_values(v)
        if isinstance(v, str):
            d[k] = string_to_tuple(v)


def string_to_tuple(
    tuple_str: str, from_list=False
) -> tuple[float, ...] | tuple[str, ...] | float | str:
    """Convert a string to a tuple if it matches a tuple pattern.

    Parameters
    ----------
    tuple_str : str
        A string representing some tuple to convert
        the numbers inside the string tuple are mapped to float
    from_list : bool, optional
        only if true will a single number string be converted to float,
        otherwise returned as-is since it may represent a label,
        by default False

    Returns
    -------
    tuple[float], tuple[str], float, str
        Returns the tuple intended by the string
    """
    if rp.tuple_number.match(tuple_str):
        return tuple(map(float, rp.number.findall(tuple_str)))
    elif rp.tuple_word.match(tuple_str):
        return tuple(map(str, rp.word.findall(tuple_str)))
    elif from_list and rp.number.match(tuple_str):
        return float(tuple_str)
    else:
        return tuple_str


def list_string_to_tuple(
    a_list: list[str],
) -> list[tuple[float, ...] | tuple[str, ...] | float | str]:
    """Convert a list of strings (representing tuples) to a list of tuples.

    Parameters
    ----------
    a_list : List[str]
        A list of strings, some of them representing (numbered) tuples

    Returns
    -------
    List[Union[float, str]]
        A list of the (numbered) tuples represted by the incoming a_list
    """
    return [string_to_tuple(v, from_list=True) for v in a_list]


def sanitize_yaml(d: dict, do_keys: bool = True, do_values: bool = False) -> dict:
    """Sanitize a yaml-returned dict for key or (list) values containing tuples.

    Parameters
    ----------
    d : dict
        a dict resulting from parsing a pyglotaran model spec yml file
    do_keys : bool
        toggle sanitization of dict keys, by default True
    do_values : bool
        toggle sanitization of dict values, by default False

    Returns
    -------
    dict
        a sanitized dict with (broken) string tuples restored as proper tuples
    """
    if do_keys:
        sanitize_dict_keys(d)
    if do_values:
        # this is only needed to allow for tuple parsing in specification
        sanitize_dict_values(d)
    return d


def sanitize_parameter_list(parameter_list: list[str | float]) -> list[str | float]:
    """Replace in a list strings matching scientific notation with floats.

    Parameters
    ----------
    parameter_list : list
        A list of parameters where some elements may be strings like 1E7

    Returns
    -------
    list
        A list where strings matching a scientific number have been converted to float
    """
    for i, value in enumerate(parameter_list):
        if isinstance(value, str) and rp.number_scientific.match(value):
            parameter_list[i] = float(value)

    return parameter_list


def check_deprecations(spec: dict):
    """Check deprecations in a `spec` dict.

    Parameters
    ----------
    spec : dict
        A specification dictionary
    """
    if "type" in spec:
        if spec["type"] == "kinetic-spectrum":
            warn_deprecated(
                deprecated_qual_name_usage="type: kinectic-spectrum",
                new_qual_name_usage="default-megacomplex: decay",
                to_be_removed_in_version="0.6.0",
                check_qual_names=(False, False),
            )
            spec["default-megacomplex"] = "decay"
        elif spec["type"] == "spectral":
            warn_deprecated(
                deprecated_qual_name_usage="type: spectral",
                new_qual_name_usage="default-megacomplex: spectral",
                to_be_removed_in_version="0.6.0",
                check_qual_names=(False, False),
            )
            spec["default-megacomplex"] = "spectral"
        del spec["type"]

    if "spectral_relations" in spec:
        warn_deprecated(
            deprecated_qual_name_usage="spectral_relations",
            new_qual_name_usage="relations",
            to_be_removed_in_version="0.6.0",
            check_qual_names=(False, False),
        )
        spec["relations"] = spec["spectral_relations"]
        del spec["spectral_relations"]

        for i, relation in enumerate(spec["relations"]):
            if "compartment" in relation:
                warn_deprecated(
                    deprecated_qual_name_usage="relation.compartment",
                    new_qual_name_usage="relation.source",
                    to_be_removed_in_version="0.6.0",
                    check_qual_names=(False, False),
                )
                relation["source"] = relation["compartment"]
                del relation["compartment"]

    if "spectral_constraints" in spec:
        warn_deprecated(
            deprecated_qual_name_usage="spectral_constraints",
            new_qual_name_usage="constraints",
            to_be_removed_in_version="0.6.0",
            check_qual_names=(False, False),
        )
        spec["constraints"] = spec["spectral_constraints"]
        del spec["spectral_constraints"]

        for i, constraint in enumerate(spec["constraints"]):
            if "compartment" in constraint:
                warn_deprecated(
                    deprecated_qual_name_usage="constraint.compartment",
                    new_qual_name_usage="constraint.target",
                    to_be_removed_in_version="0.6.0",
                    check_qual_names=(False, False),
                )
                constraint["target"] = constraint["compartment"]
                del constraint["compartment"]

    if "equal_area_penalties" in spec:
        warn_deprecated(
            deprecated_qual_name_usage="equal_area_penalties",
            new_qual_name_usage="clp_area_penalties",
            to_be_removed_in_version="0.6.0",
            check_qual_names=(False, False),
        )
        spec["clp_area_penalties"] = spec["equal_area_penalties"]
        del spec["equal_area_penalties"]
