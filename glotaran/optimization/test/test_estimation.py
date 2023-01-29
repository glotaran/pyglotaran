from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from glotaran.model import ClpRelation
from glotaran.model import ZeroConstraint
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.test.models import TestDataModelConstantIndexDependent
from glotaran.optimization.test.models import TestDataModelConstantIndexIndependent
from glotaran.optimization.test.models import TestDataModelConstantThreeCompartments
from glotaran.parameter import Parameter


def test_estimate_dataset():
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    data = OptimizationData(data_model)
    matrix = OptimizationMatrix.from_data(data)
    matrices = [matrix.at_index(i).reduce(index, [], []) for i, index in range(data.global_axis)]
