import re
from typing import Any
from typing import List

# tuple_pattern = re.compile(r"(\(.*?,.*?\))")
tuple_number_pattern = re.compile(r"(\([\s\d.+-]+?[,\s\d.+-]*?\))")
number_pattern = re.compile(r"[\d.+-]+")
tuple_name_pattern = re.compile(r"(\([.\s\w\d]+?[,.\s\w\d]*?\))")
name_pattern = re.compile(r"[\w]+")
group_pattern = re.compile(r"(\(.+?\))")
match_list_with_tuples = re.compile(r"(\[.+\(.+\).+\])")
match_elements_in_string_of_list = re.compile(r"(\(.+?\)|[-+.\d]+)")


def sanitize_list_with_broken_tuples(mangled_list: List[Any]) -> List[str]:
    """Sanitize a list with 'broken' tuples

    A list of broken tuples as returned by yaml when parsing tuples.
    e.g parsing the list of tuples [(3,100), (4,200)] results in
    a list of str ['(3', '100)', '(4', '200)'] which can be restored to
    a list with the tuples restored as strings ['(3, 100)', '(4, 200)']

    Args:
        mangled_list (List[str,float]): [description]

    Returns:
        List[str]: [description]
    """
    sanitized_string = str(mangled_list).replace("'", "")
    return list(match_elements_in_string_of_list.findall(sanitized_string))


def sanitize_dict_keys(d):
    if not isinstance(d, (dict, list)):
        return
    d_new = {}
    for k, v in d.items() if isinstance(d, dict) else enumerate(d):
        if isinstance(d, dict) and isinstance(k, str) and tuple_name_pattern.match(k):
            k_new = tuple(map(str, name_pattern.findall(k)))
            d_new.update({k_new: v})
        elif isinstance(d, (dict, list)):
            new_v = sanitize_dict_keys(v)
            if new_v:
                d[k] = new_v
    return d_new


def sanitize_dict_values(d):
    if not isinstance(d, (dict, list)):
        return
    for k, v in d.items() if isinstance(d, dict) else enumerate(d):
        if isinstance(v, list):
            leaf = all(isinstance(el, (str, tuple, float)) for el in v)
            if leaf:
                # print(f"is_leaf: {v}")
                if "(" in str(v):
                    d[k] = list_to_tuple(sanitize_list_with_broken_tuples(v))
                # print(d)
            else:
                sanitize_dict_values(v)
        if isinstance(v, dict):
            sanitize_dict_values(v)
        if isinstance(v, str):
            d[k] = str_to_tuple(v)


def str_to_tuple(v, from_list=False):
    if tuple_number_pattern.match(v):
        return tuple(map(float, number_pattern.findall(v)))
    elif tuple_name_pattern.match(v):
        return tuple(map(str, name_pattern.findall(v)))
    elif from_list and number_pattern.match(v):
        return float(v)
    else:
        return v


def list_to_tuple(a_list):
    for i, v in enumerate(a_list):
        a_list[i] = str_to_tuple(v, from_list=True)
    return a_list


def sanitize_yaml(d):
    """Sanitize a yaml-returned dict for key or (list) values containing tuples

    Args:
        d (dict): a dict resulting from parsing a pyglotaran model spec yml file

    Returns:
        dict: a sanitized dict with (broken) string tuples restored as proper tuples
    """
    sanitize_dict_keys(d)
    # sanitize_dict_values(d)
    return d
