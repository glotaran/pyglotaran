"""Functions for reading and parsing models from serialized representations."""

from typing import Dict

import yaml

from glotaran.parse.util import sanitize_yaml

from .register import get_model
from .register import known_model


def parse_yaml_file(filename: str) -> Dict:
    """parse_yaml_file reads the given file and parses its content as YML.

    Parameters
    ----------
    filename : str
        filename is the of the file to parse.

    Returns
    -------
    content : Dict
        The content of the file as dictionary.
    """

    with open(filename) as f:
        spec = parse_yaml(f)

    return spec


def parse_yaml(data: str):
    try:
        return yaml.safe_load(data)
    except Exception as e:
        raise e


def parse_spec(spec: Dict):

    spec = sanitize_yaml(spec)

    if "type" not in spec:
        raise Exception("Model type not defined")

    model_type = spec["type"]
    del spec["type"]

    if not known_model(model_type):
        raise Exception(f"Unknown model type '{model_type}'.")

    model = get_model(model_type)
    try:
        return model.from_dict(spec)
    except Exception as e:
        raise e


def load_yaml_file(filename: str):
    spec = parse_yaml_file(filename)
    return parse_spec(spec)


def load_yaml(data: str):
    spec = parse_yaml(data)
    return parse_spec(spec)
