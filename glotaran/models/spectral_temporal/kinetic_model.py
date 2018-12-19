"""Glotaran Kinetic Model"""

import numpy as np

from glotaran.model import model, BaseModel

from .initial_concentration import InitialConcentration
from .irf import Irf
from .k_matrix import KMatrix
from .kinetic_fit_result import KineticFitResult
from .kinetic_megacomplex import KineticMegacomplex
from .spectral_constraints import (
    SpectralConstraint,
    OnlyConstraint,
    ZeroConstraint,
    EqualAreaConstraint,
)
from .spectral_relations import SpectralRelation
from .spectral_shape import SpectralShape
from .spectral_temporal_dataset_descriptor import SpectralTemporalDatasetDescriptor
from .kinetic_matrix import calculate_kinetic_matrix
from .spectral_matrix import calculate_spectral_matrix


def apply_spectral_constraints_and_relations(model, index, clp_labels, matrix):

    for constraint in model.spectral_constraint:

        if constraint.applies(index):

            if isinstance(constraint, (OnlyConstraint, ZeroConstraint)):
                idx = clp_labels.index(constraint.compartment)
                del clp_labels[idx]

                matrix = np.delete(matrix, idx, axis=1)

    for relation in model.spectral_relations:
        if relation.applies(index):
            idx = clp_labels.index(relation.compartment)
            target = clp_labels.index(relation.target)
            del clp_labels[idx]

            matrix[:, target] += relation.parameter * matrix[:, idx]
            matrix = np.delete(matrix, idx, axis=1)

    return clp_labels, matrix


def apply_equality_constraints(model, clp_labels, clp, concentrations):
    for constraint in model.spectral_constraint:
            residuals = []
            if isinstance(constraint, EqualAreaConstraint):
                residual = []
                for index in clp:
                    if constraint.applies(index):
                        labels = clp_labels[index]
                        idx = labels.index(constraint.compartment)
                        target = labels.index(constraint.target)
                        residual.append(
                            np.abs(
                                clp[index[idx] - clp[index][target]]
                            )
                        )
                residuals.append(residual)
            return residuals


@model(
    'kinetic',
    attributes={
        'initial_concentration': InitialConcentration,
        'k_matrix': KMatrix,
        'irf': Irf,
        'shape': SpectralShape,
        'spectral_constraints': SpectralConstraint,
        'spectral_relations': SpectralRelation,
    },
    dataset_type=SpectralTemporalDatasetDescriptor,
    megacomplex_type=KineticMegacomplex,
    calculated_matrix=calculate_kinetic_matrix,
    calculated_axis='time',
    estimated_matrix=calculate_spectral_matrix,
    estimated_axis='spectral',
    fit_result_class=KineticFitResult
)
class KineticModel(BaseModel):
    """
    A kinetic model is an implementation for model.Model. It is used describe
    time dependend datasets.
    """
