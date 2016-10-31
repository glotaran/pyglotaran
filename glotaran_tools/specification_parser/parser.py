import yaml
import os
from .model_spec_yaml import get_model_parser


def parse_file(fname):
    if not os.path.isfile(fname):
        raise Exception("File does not exist.")

    f = open(fname)
    spec = load(f)
    f.close

    parser = get_model_parser(spec)

    parser.parse()

    return parser.model


def load(s):
    return yaml.load(s)
