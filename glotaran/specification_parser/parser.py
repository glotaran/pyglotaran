import yaml
import os
from .model_spec_yaml import get_model_parser


def parse_file(fname):
    if not os.path.isfile(fname):
        raise Exception("File does not exist.")

    with open(fname) as f:
        spec = load(f)

    return parse_spec(spec)


def parse_yml(spec):
    spec = load(spec)
    return parse_spec(spec)


def parse_spec(spec):

    parser = get_model_parser(spec)

    parser.parse()

    return parser.model


def load(s):
    return yaml.load(s)
