import abc
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from glotaran.model import ClpConstraint
from glotaran.model import ClpPenalty
from glotaran.model import ClpRelation
from glotaran.model import DataModel
from glotaran.model import ExperimentModel
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.penalty import calculate_clp_penalties


class OptimizationObjective:
    def __init__(
        self,
        global_axis: ArrayLike,
        constraints: list[ClpConstraint],
        relations: list[ClpRelation],
        penalties: list[ClpPenalty],
        residual_function: Literal["variable_projection", "non_negative_least_squares"],
    ):
        self._global_axis = global_axis
        self._constraints = constraints
        self._relations = relations
        self._penalties = penalties
        self._residual_function = residual_function

    @abc.abstractmethod
    def calculate_matrices(self) -> list[OptimizationMatrix]:
        pass

    @abc.abstractmethod
    def get_data(self, index: int) -> ArrayLike:
        pass

    def calculate(self) -> ArrayLike:
        matrices = self.calculate_matrices()
        reduced_matrices = [
            matrices[i].reduce(index, self._constraints, self._relations)
            for i, index in enumerate(self._global_axis)
        ]
        estimations = [
            OptimizationEstimation.calculate(
                matrices[i].array, self.get_data(i), self._residual_function
            )
            for i in range(self._global_axis.size)
        ]

        penalties = [e.residual for e in estimations]
        if len(self._penalties) > 0:
            estimations = [
                e.resolve_clp(m.clp_labels, r.clp_labels, i, self._constraints, self._relations)
                for e, m, r, i in zip(estimations, matrices, reduced_matrices, self._global_axis)
            ]
            penalties += calculate_clp_penalties(
                matrices, estimations, self._global_axis, self._penalties
            )
        return np.concatenate(penalties)


class OptimizationObjectiveData(OptimizationObjective):
    def __init__(
        self,
        model: DataModel,
        constraints: list[ClpConstraint],
        relations: list[ClpRelation],
        penalties: list[ClpPenalty],
    ):
        self._data = OptimizationData(model)
        super.__init__(
            self._data.global_axis, constraints, relations, penalties, model.residual_function
        )

    def calculate_matrices(self) -> list[OptimizationMatrix]:
        matrix = OptimizationMatrix.from_data(self._data)
        return [matrix.at_index(i) for i in range(self._data.global_axis.size)]

    def get_data(self, index: int) -> ArrayLike:
        return self._data.data[:, index]


class OptimizationObjectiveExperiment:
    def __init__(
        self,
        model: ExperimentModel,
    ):
        self._data = LinkedOptimizationData(
            model.datasets, model.clp_link_tolerance, model.clp_link_method, model.scale
        )
        super.__init__(
            self._data.global_axis,
            model.clp_constraints,
            model.clp_relations,
            model.clp_penalties,
            model.residual_function,
        )

    def calculate_matrices(self) -> list[OptimizationMatrix]:
        return OptimizationMatrix.from_linked_data(self._data)

    def get_data(self, index: int) -> ArrayLike:
        return self._data.data[index]

    def calculate(self) -> ArrayLike:
        matrices = OptimizationMatrix.from_linked_data(self._data)
        reduced_matrices = [
            matrices[i].reduce(index, self._constraints, self._relations)
            for i, index in enumerate(self._data.global_axis)
        ]
        estimations = [
            OptimizationEstimation.calculate(
                matrices[i].array, self._data.data[:, i], self._data.model.residual_function
            )
            for i in range(self._data.global_axis.size)
        ]

        penalties = [e.residual for e in estimations]
        if len(self._penalties) > 0:
            estimations = [
                e.resolve_clp(m.clp_labels, r.clp_labels, i, self._constraints, self._relations)
                for e, m, r, i in zip(
                    estimations, matrices, reduced_matrices, self._data.global_axis
                )
            ]
            penalties += calculate_clp_penalties(
                matrices, estimations, self._data.global_axis, self._penalties
            )
        return np.concatenate(penalties)
