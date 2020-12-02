from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.result import Result
from glotaran.analysis.scheme import Scheme
from glotaran.analysis.simulation import simulate
from glotaran.parameter import ParameterGroup

from .dataset_descriptor import DatasetDescriptor
from .decorator import FinalizeFunction
from .weight import Weight

Cls = TypeVar("Cls")


class Model:
    _model_type: str
    dataset: Mapping[str, DatasetDescriptor]
    megacomplex: Any
    weights: Weight
    model_dimension: str
    global_dimension: str
    global_matrix = None
    finalize_data: Optional[FinalizeFunction] = None
    grouped: Callable[[Type[Model]], bool]
    index_dependent: Callable[[Type[Model]], bool]

    @staticmethod
    def matrix(
        dataset_descriptor: DatasetDescriptor = None, axis=None, index=None
    ) -> Union[Tuple[None, None], Tuple[List[Any], np.ndarray]]:
        ...

    def add_megacomplex(self, item: Any):
        ...

    def add_weights(self, item: Weight):
        ...

    def get_dataset(self, label: str) -> DatasetDescriptor:
        ...

    @classmethod
    def from_dict(cls: Type[Cls], model_dict_ref: Dict) -> Cls:
        ...

    @property
    def index_depended_matrix(self):
        ...

    @property
    def model_type(self) -> str:
        ...

    def simulate(
        self,
        dataset: str,
        parameter: ParameterGroup,
        axes: Dict[str, np.ndarray] = ...,
        clp: Union[np.ndarray, xr.DataArray] = ...,
        noise: bool = ...,
        noise_std_dev: float = ...,
        noise_seed: int = ...,
    ) -> xr.Dataset:
        ...

    def result_from_parameter(
        self,
        parameter: ParameterGroup,
        data: Dict[str, Union[xr.DataArray, xr.Dataset]],
        nnls: bool = ...,
        group_atol: float = ...,
    ) -> Result:
        ...

    def problem_list(self, parameter: ParameterGroup = ...) -> List[str]:
        ...

    def validate(self, parameter: ParameterGroup = ...) -> str:
        ...

    def valid(self, parameter: ParameterGroup = None) -> bool:
        ...

    def markdown(self, parameter: ParameterGroup = ..., initial: ParameterGroup = ...) -> str:
        ...
