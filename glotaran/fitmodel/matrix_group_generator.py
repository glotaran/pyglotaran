from collections import OrderedDict
import numpy as np

from .matrix_group import MatrixGroup


class MatrixGroupGenerator(object):
    def __init__(self, matrix, calculated=False):
        self._groups = OrderedDict()
        self._matrix = matrix
        self._calculated = calculated

    @classmethod
    def for_model(cls, model, matrix, xtol=0.5, calculated=False):
        gen = cls(matrix, calculated)
        gen._init_groups_for_model(model, xtol)
        return gen

    @classmethod
    def for_dataset(cls, model, dataset, matrix, calculated=False):
        gen = cls(matrix, calculated)
        data = model.datasets[dataset]
        gen._add_dataset_to_group(model, data, 0)
        return gen

    def _init_groups_for_model(self, model, xtol):
        for _, dataset in model.datasets.items():
            self._add_dataset_to_group(model, dataset, xtol)

    def _add_dataset_to_group(self, model, dataset, xtol):
        grouping_axis = dataset.data.get_calculated_axis() if self._calculated\
                else dataset.data.get_estimated_axis()
        for matrix in [self._matrix(x, dataset, model) for x
                       in grouping_axis]:
            self._add_c_matrix_to_group(matrix, xtol)

    def _add_c_matrix_to_group(self, matrix, xtol):
                if matrix.x in self._groups:
                    self._groups[matrix.x].add_cmatrix(matrix)
                elif any(abs(matrix.x-val) < xtol for val in self._groups):
                    idx = [val for val in self._groups if abs(matrix.x-val) <
                           xtol][0]
                    self._groups[idx].add_cmatrix(matrix)
                else:
                    self._groups[matrix.x] = MatrixGroup(matrix)

    def groups(self):
        for _, group in self._groups.items():
            yield group

    def calculate(self, parameter):
        return [group.calculate(parameter) for group in self.groups()]

    def create_dataset_group(self):

        dataset_group = []
        for _, group in self._groups.items():
            slice = np.array([])
            for mat in group.c_matrices:
                x = np.where(mat.dataset.data.get_estimated_axis() ==
                             mat.x)
                slice = np.concatenate((slice,
                                        mat.dataset.data.data[x, :].flatten()))
            dataset_group.append(slice)
        return dataset_group


def calc(data):
    (res, parameter, group, index) = data
    res[index] = group.calculate(parameter)
