from glotaran_core import (constraints, dataset, global_analysis, model,
                           result, kinetic_model, parameter)


Scaling = dataset.Scaling
Model = model.Model
Result = result.Result
GlobalAnalysis = global_analysis.GlobalAnalysis

KineticModel = kinetic_model.KineticModel
KineticDataset = kinetic_model.KineticDataset
KineticMegacomplex = kinetic_model.KineticMegacomplex
KMatrix = kinetic_model.KMatrix
GaussianIrf = kinetic_model.GaussianIrf

Parameter = parameter.Parameter
create_parameter_list = parameter.create_parameter_list

ZeroConstraint = constraints.ZeroConstraint
EqualConstraint = constraints.EqualConstraint
EqualAreaConstraint = constraints.EqualAreaConstraint

FixedConstraint = constraints.FixedConstraint
BoundConstraint = constraints.BoundConstraint

Relation = constraints.Relation
