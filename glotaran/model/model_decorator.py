from collections import OrderedDict
from typing import List, Type, Dict, Generator

from glotaran.fitmodel import FitModel, Result
from glotaran.parse.register import register_model

from .dataset_descriptor import DatasetDescriptor
from .initial_concentration import InitialConcentration
from .megacomplex import Megacomplex


def glotaran_model(name,
                   attributes={},
                   dataset_type=DatasetDescriptor,
                   fitmodel_type=FitModel,
                   megacomplex_type=Megacomplex,
                   calculated_matrix=None,
                   estimated_matrix=None,
                   calculated_axis=None,
                   estimated_axis=None,
                   ):

    def decorator(cls):

        setattr(cls, 'model_type', name)
        setattr(cls, 'dataset_type', dataset_type)
        setattr(cls, 'fitmodel_type', fitmodel_type)

        def c_mat(self, c_mat=calculated_matrix):
            return c_mat
        setattr(cls, 'calculated_matrix', property(c_mat))
        setattr(cls, 'calculated_axis', calculated_axis)

        def e_mat(self, e_mat=estimated_matrix):
            return e_mat
        setattr(cls, 'estimated_matrix', property(e_mat))
        setattr(cls, 'estimated_axis', estimated_axis)

        if not hasattr(cls, '__annotations__'):
            setattr(cls, '__annotations__', {})

        if not hasattr(cls, '_glotaran_model_attributes'):
            setattr(cls, '_glotaran_model_attributes', {})

        # Set annotations and methods for attributes

        # Add standart attributes if not present
        if 'dataset' not in getattr(cls, '__annotations__') or \
                dataset_type is not DatasetDescriptor:
            attributes['dataset'] = dataset_type
        if 'initial_concentration' not in getattr(cls, '__annotations__'):
            attributes['initial_concentration'] = InitialConcentration
        if 'megacomplex' not in getattr(cls, '__annotations__') \
                or megacomplex_type is not Megacomplex:
            attributes['megacomplex'] = megacomplex_type

        for attr_name, attr_type in attributes.items():
            getattr(cls, '__annotations__')[attr_name] = Dict[str, attr_type]
            getattr(cls, '_glotaran_model_attributes')[attr_name] = None

            def get_item(self, label: str, attr_name=attr_name):
                return getattr(self, attr_name)[label]

            setattr(cls, f"get_{attr_name}", get_item)

            def set_item(self, label: str, item: attr_type,
                         attr_name=attr_name,
                         attr_type=attr_type):

                # TODO checked typed items
                if not isinstance(item, attr_type) and \
                        not hasattr(attr_type, "_glotaran_model_item_typed"):
                    raise TypeError
                getattr(self, attr_name)[label] = item

            setattr(cls, f"set_{attr_name}", set_item)

        def init(self, cls=cls, attributes=attributes):
            for attr_name in attributes:
                setattr(self, attr_name, OrderedDict())
            super(cls, self).__init__()

        setattr(cls, '__init__', init)

        register_model(name, cls)

        return cls

    return decorator
