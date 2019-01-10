from collections import OrderedDict
from typing import Dict, List

from glotaran.parse.register import register_model

from .dataset_descriptor import DatasetDescriptor


def model(name,
          attributes={},
          dataset_type=DatasetDescriptor,
          megacomplex_type=None,
          calculated_matrix=None,
          estimated_matrix=None,
          calculated_axis=None,
          estimated_axis=None,
          constrain_calculated_matrix_function=None,
          additional_residual_function=None,
          finalize_result_function=None,
          allow_grouping=True,
          ):

    def decorator(cls):

        setattr(cls, 'model_type', name)
        setattr(cls, 'dataset_type', dataset_type)
        setattr(cls, 'finalize_result', finalize_result_function)
        setattr(cls, 'constrain_calculated_matrix_function',
                constrain_calculated_matrix_function)
        setattr(cls, 'additional_residual_function',
                additional_residual_function)
        setattr(cls, 'allow_grouping', allow_grouping)

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
        else:
            setattr(cls, '__annotations__',
                    getattr(cls, '__annotations__').copy())

        if not hasattr(cls, '_glotaran_model_attributes'):
            setattr(cls, '_glotaran_model_attributes', {})
        else:
            setattr(cls, '_glotaran_model_attributes',
                    getattr(cls, '_glotaran_model_attributes').copy())

        # Add standard attributes if not present
        attributes['megacomplex'] = megacomplex_type
        attributes['dataset'] = dataset_type

        # Set annotations and methods for attributes

        for attr_name, attr_type in attributes.items():
            if getattr(attr_type, '_glotaran_has_label'):
                getattr(cls, '__annotations__')[attr_name] = Dict[str, attr_type]
            else:
                getattr(cls, '__annotations__')[attr_name] = List[attr_type]
            getattr(cls, '_glotaran_model_attributes')[attr_name] = None

            if getattr(attr_type, '_glotaran_has_label'):

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
                setattr(cls, attr_name, {})

            else:
                def add_item(self, item: attr_type,
                             attr_name=attr_name,
                             attr_type=attr_type):

                    # TODO checked typed items
                    if not isinstance(item, attr_type) and \
                            not hasattr(attr_type, "_glotaran_model_item_typed"):
                        raise TypeError
                    getattr(self, attr_name).append(item)

                setattr(cls, f"add_{attr_name}", add_item)
                setattr(cls, attr_name, [])

        def init(self, cls=cls, attributes=attributes):
            for attr_name, attr_item in attributes.items():
                if getattr(attr_item, '_glotaran_has_label'):
                    setattr(self, attr_name, OrderedDict())
                else:
                    setattr(self, attr_name, [])
            super(cls, self).__init__()

        setattr(cls, '__init__', init)

        register_model(name, cls)

        return cls

    return decorator
