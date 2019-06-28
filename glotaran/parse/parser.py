"""Functions for reading and parsing models from serialized representations."""

from typing import Dict
import yaml
import re
from .register import get_model, known_model


# shamelessly taken from
# https://stackoverflow.com/questions/39553008/how-to-read-a-python-tuple-using-pyyaml#39553138
# this is to convert the string written as a tuple into a python tuple
def _yml_tuple_constructor(loader, node):
    # this little parse is really just for what I needed, feel free to change it!
    def parse_tup_el(el):
        # try to convert into int or float else keep the string
        if el.isdigit():
            return int(el)
        try:
            return float(el)
        except ValueError:
            return el

    value = loader.construct_scalar(node)
    # remove the ( ) from the string
    tup_elements = [ele.strip() for ele in value[1:-1].split(',')]
    # remove the last element if the tuple was written as (x,b,)
    if tup_elements[-1] == '':
        tup_elements.pop(-1)
    tup = tuple(map(parse_tup_el, tup_elements))
    return tup


# !tuple is my own tag name, I think you could choose anything you want
yaml.FullLoader.add_constructor(u'!tuple', _yml_tuple_constructor)
# this is to spot the strings written as tuple in the yaml
yaml.FullLoader.add_implicit_resolver(u'!tuple', re.compile(r"\((.*?,.*?)\)"), None)


def parse_yml_file(fname: str) -> Dict:
    """parse_yml_file reads the given file and parses its content as YML.

    Parameters
    ----------
    fname : str
        fname is the of the file to parse.

    Returns
    -------
    content : Dict
        The content of the file as dictionary.
    """

    with open(fname) as f:
        spec = parse_yml(f)

    return spec


def parse_yml(data: str):
    try:
        return yaml.load(data, Loader=yaml.FullLoader)
    except Exception as e:
        raise e


def parse_spec(spec: Dict):

    if 'type' not in spec:
        raise Exception("Model type not defined")

    model_type = spec['type']
    del spec['type']

    if not known_model(model_type):
        raise Exception(f"Unknown model type '{model_type}'.")

    model = get_model(model_type)
    try:
        return model.from_dict(spec)
    except Exception as e:
        raise e


def load_yml_file(fname: str):
    spec = parse_yml_file(fname)
    return parse_spec(spec)


def load_yml(data: str):
    spec = parse_yml(data)
    return parse_spec(spec)
