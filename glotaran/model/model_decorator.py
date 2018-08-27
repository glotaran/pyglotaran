from .dataset_descriptor import DatasetDescriptor
from .initial_concentration import InitialConcentration
from .megacomplex import Megacomplex
from glotaran.fitmodel import FitModel, Matrix, Result
from collections import OrderedDict
from typing import List, Type, Dict, Generator

def glotaran_model(name,
                   attributes={},
                   dataset_type=DatasetDescriptor,
                   fitmodel_type=FitModel,
                   megacomplex_type=Megacomplex,
                   calculated_matrix=Matrix,
                   estimated_matrix=Matrix,
                   ):

    def decorator(cls):

        setattr(cls, 'model_type', name)
        setattr(cls, 'dataset_type', dataset_type)
        setattr(cls, 'fitmodel_type', fitmodel_type)
        setattr(cls, 'calculated_matrix', calculated_matrix)
        setattr(cls, 'estimated_matrix', estimated_matrix)

        setattr(cls, '__annotations__', {})

        # Set annotations and methods for attributes

        # Add standart attributes
        attributes['dataset'] = dataset_type
        attributes['initial_concentration'] = InitialConcentration
        attributes['megacomplex'] = megacomplex_type

        for attr_name, attr_type in attributes.items():
            getattr(cls, '__annotations__')[attr_name] = Dict[str, attr_type]

            def get_item(self, label: str, attr_name=attr_name):
                return getattr(self, attr_name)[label]

            setattr(cls, f"get_{attr_name}", get_item)

            def set_item(self, label: str, item: attr_type,
                         attr_name=attr_name,
                         attr_type=attr_type):
                getattr(cls, '__annotations__')[attr_name] = Dict[str, attr_type]
                print(attr_name)
                if not isinstance(item, attr_type):
                    raise TypeError
                getattr(self, attr_name)[label] = item

            setattr(cls, f"set_{attr_name}", set_item)

        def init(self, cls=cls, attributes=attributes):
            for attr_name in attributes:
                setattr(cls, attr_name, OrderedDict())
            super(cls, self).__init__()

        setattr(cls, '__init__', init)

        def string(self, attributes=attributes):
            string = "# Model\n\n"
            string += f"_Type_: {self.model_type}\n\n"

            for attr in attributes:
                string += f"## {attr}\n"

                for label, item in getattr(self, attr).items():
                    string += f'{item}\n'
            return string

        setattr(cls, '__str__', string)

        return cls

    return decorator
