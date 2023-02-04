from __future__ import annotations

from glotaran.builtin.items.activation.activation import Activation
from glotaran.model import Attribute
from glotaran.model import DataModel
from glotaran.model import ItemIssue
from glotaran.model import Library
from glotaran.parameter import Parameters


class NoActivationIssue(ItemIssue):
    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return "ActivationError: No Activation defined in dataset."


def validate_activations(
    value: list[Activation],
    activation: Activation,
    library: Library,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    issues = []
    if len(value) == 0:
        issues.append(NoActivationIssue())
    return issues


class ActivationDataModel(DataModel):
    activation: list[Activation] = Attribute(
        validator=validate_activations,
        description="The activation(s) of the dataset.",
    )
