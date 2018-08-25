"""Glotarans model package"""
from . import (
               compartment_constraints,
               dataset,
               dataset_descriptor,
               initial_concentration,
               megacomplex,
               model,
               parameter,
               parameter_group,
               )


# Compartment Constraints
CompartmentConstraintType = compartment_constraints.CompartmentConstraintType
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

def glotaran_attribute(attrib):
    name = attrib.__name__

    def add_attrib(model, items):
        setattr(model, name, items)

    Model._attributes[name] = add_attrib
    return attrib
