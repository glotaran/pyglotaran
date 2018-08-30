"""Glotarans model package"""
ModelError = None
ParameterGroup = None

from . import (
    compartment_constraints,
    dataset,
    dataset_descriptor,
    initial_concentration,
    megacomplex,
    model,
    model_decorator,
    model_item,
    parameter,
    parameter_group,
)


# Compartment Constraints
CompartmentConstraint = compartment_constraints.CompartmentConstraint
ZeroConstraint = compartment_constraints.ZeroConstraint
EqualConstraint = compartment_constraints.EqualConstraint
EqualAreaConstraint = compartment_constraints.EqualAreaConstraint

# Dataset

DatasetDescriptor = dataset_descriptor.DatasetDescriptor
Dataset = dataset.Dataset

# Initial Concentration

InitialConcentration = initial_concentration.InitialConcentration

# Megacomplex

Megacomplex = megacomplex.Megacomplex

# Model

Model = model.Model

# Parameter

Parameter = parameter.Parameter
ParameterGroup = parameter_group.ParameterGroup


# Decorators

glotaran_model = model_decorator.glotaran_model
glotaran_model_item = model_item.glotaran_model_item
glotaran_model_item_typed = model_item.glotaran_model_item_typed


class ModelError(Exception):
    def __init__(self, model):

        msg = "Model Error\n"
        msg += "-----------\n\n"
        msg += "Please fix the following issues:\n"
        for error in model.errors():
            msg += f"* {error}\n"

        # Call the base class constructor with the parameters it needs
        super().__init__(msg)

