"""Glotaran Kinetic Model"""

import typing
import numpy as np

from glotaran.model import model, Model
from glotaran.parameter import ParameterGroup

from .initial_concentration import InitialConcentration
from .irf import Irf
from .k_matrix import KMatrix
from .kinetic_fit_result import finalize_kinetic_result
from .kinetic_megacomplex import KineticMegacomplex
from .spectral_constraints import (
    SpectralConstraint, OnlyConstraint, ZeroConstraint, EqualAreaConstraint)
from .spectral_relations import SpectralRelation
from .spectral_shape import SpectralShape
from .spectral_temporal_dataset_descriptor import SpectralTemporalDatasetDescriptor
from .kinetic_matrix import calculate_kinetic_matrix
from .spectral_matrix import calculate_spectral_matrix


def apply_spectral_constraints(
        model: typing.Type['KineticModel'],
        clp_labels: typing.List[str],
        matrix: np.ndarray,
        index: float):
    for constraint in model.spectral_constraints:
        if isinstance(constraint, (OnlyConstraint, ZeroConstraint)) and constraint.applies(index):
            idx = [not label == constraint.compartment for label in clp_labels]
            clp_labels = [label for label in clp_labels if not label == constraint.compartment]
            matrix = matrix[:, idx]
    return (clp_labels, matrix)


def spectral_constraint_penalty(
        model: typing.Type['KineticModel'],
        parameter: ParameterGroup,
        clp_labels: typing.List[str],
        clp: np.ndarray,
        matrix: np.ndarray,
        index: float):
    residual = []
    for constraint in model.spectral_constraints:
        if isinstance(constraint, EqualAreaConstraint) and constraint.applies(index):
            constraint = constraint.fill(model, parameter)
            source_idx = clp_labels.index(constraint.compartment)
            target_idx = clp_labels.index(constraint.target)
            residual.append(
                (clp[source_idx] - constraint.parameter * clp[target_idx]) * constraint.weight
            )
    return residual


def apply_spectral_relations(
        model: typing.Type['KineticModel'],
        parameter: ParameterGroup,
        clp_labels: typing.List[str],
        matrix: np.ndarray,
        index: float):
    for relation in model.spectral_relations:
        if relation.applies(index):
            relation = relation.fill(model, parameter)
            source_idx = clp_labels.index(relation.compartment)
            target_idx = clp_labels.index(relation.target)
            matrix[:, target_idx] += relation.parameter * matrix[:, source_idx]

            idx = [not label == relation.compartment for label in clp_labels]
            clp_labels = [label for label in clp_labels if not label == relation.compartment]
            matrix = matrix[:, idx]
    return (clp_labels, matrix)


def apply_kinetic_model_constraints(
        model: typing.Type['KineticModel'],
        parameter: ParameterGroup,
        clp_labels: typing.List[str],
        matrix: np.ndarray,
        index: float):
    clp_labels, matrix = apply_spectral_relations(model, parameter, clp_labels, matrix, index)
    clp_labels, matrix = apply_spectral_constraints(model, clp_labels, matrix, index)
    return clp_labels, matrix


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
    finalize_result_function=finalize_kinetic_result,
    constrain_calculated_matrix_function=apply_kinetic_model_constraints,
    additional_penalty_function=spectral_constraint_penalty
)
class KineticModel(Model):
    """
    A kinetic model is an implementation for model.Model. It is used describe
    time dependend datasets.
    """
