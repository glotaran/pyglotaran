"""This package contains the intial concentration item."""

from typing import List

from glotaran.model import model_item


@model_item(attributes={'parameters': List[str]})
class InitialConcentration:
    """An initial concentration describes the population of the compartments at
    the beginning of an experiement."""
