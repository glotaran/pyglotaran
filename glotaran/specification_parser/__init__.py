"""Glotarans parsing package"""
from . import parser, model_spec_yaml_kinetic, model_spec_yaml_doas  # noqa: F401

parse_file = parser.parse_file
parse_yml = parser.parse_yml
