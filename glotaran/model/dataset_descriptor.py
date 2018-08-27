"""Dataset Descriptor"""

from typing import Dict, List
# unused imports
# from typing import Tuple
# import numpy as np

from .compartment_constraints import CompartmentConstraint
from .dataset import Dataset
from .model_item import glotaran_model_item


@glotaran_model_item(attributes={
                        'initial_concentration': str,
                        'megacomplexes': List[str],
                        'scale': (str, None),
                        'compartment_constraints': (List[CompartmentConstraint], None),
                     },
                     validate_model=['initial_concentration']
                     )
class DatasetDescriptor:
    """Represents a dataset for fitting

    Parameters
    ----------
    label : str
        The label of the dataset.

    initial_concentration : str
        The label of the initial concentration

    megacomplexes : List[str]
        A list of megacomplex labels

    megacomplex_scaling : Dict[str: List[str]]
        The megacomplex scaling parameters

    scaling : str
        The scaling parameter for the dataset

    compartment_scaling: Dict[str: List[str]]
        The compartment scaling parameters

    compartment_constraints: List[CompartmentConstraint] :
        A list of compartment constraints

    """
    _data = None

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=attribute-defined-outside-init
    # Datasets are complex.

    @property
    def dataset(self):
        """An implementation of model.Dataset"""
        return self._data

    @dataset.setter
    def dataset(self, data):
        if not isinstance(data, Dataset) and data is not None:
            raise TypeError
        self._data = data
