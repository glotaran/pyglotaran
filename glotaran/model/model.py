from collections import OrderedDict
from typing import Dict, List
import functools

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
        cls.__doc__ += '''
        
        Attributes
        ----------
        allow_grouping:
            Indicates if the model is allowed to group data along the estimated_axis.

        '''

        def c_mat(self, c_mat=calculated_matrix):
            return c_mat
        setattr(cls, 'calculated_matrix', property(c_mat))
        setattr(cls, 'calculated_axis', calculated_axis)

        def e_mat(self, e_mat=estimated_matrix):
            return e_mat
        setattr(cls, 'estimated_matrix', property(e_mat))
        setattr(cls, 'estimated_axis', estimated_axis)

        if not hasattr(cls, '__annotations__'):
            setattr(cls, '__annotations__', {
                'allow_grouping': bool,
            })
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

                get_item = _create_get_func(cls, attr_name, attr_type)
                setattr(cls, get_item.__name__, get_item)

                set_item = _create_set_func(cls, attr_name, attr_type)
                setattr(cls, set_item.__name__, set_item)
                setattr(cls, attr_name, {})

            else:

                add_item = _create_add_func(cls, attr_name, attr_type)
                setattr(cls, add_item.__name__, add_item)
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


def _create_add_func(cls, name, type):

    def add_item(self, item):

        # TODO checked typed items
        if not isinstance(item, type) and \
                not hasattr(type, "_glotaran_model_item_typed"):
            raise TypeError
        getattr(self, name).append(item)

    add_item.__annotations__ = {
        'item': type,
    }
    add_item.__name__ = f'add_{name}'
    add_item.__qualname__ = cls.__name__ + '.' + add_item.__name__
    add_item.__module__ = cls.__module__
    add_item.__doc__ = f'''
    Adds an `{type.__name__}` object.

    Parameters
    ----------
    item :
        The `{type.__name__}` item.
    '''

    return add_item


def _create_get_func(cls, name, type):

    def get_item(self, label):
        return getattr(self, name)[label]

    get_item.__annotations__ = {
        'label': str,
        'return': type,
    }
    get_item.__name__ = f'get_{name}'
    get_item.__qualname__ = cls.__qualname__ + '.' + get_item.__name__
    get_item.__module__ = cls.__module__
    get_item.__doc__ = f'''
    Returns the `{type.__name__}` object with the given label.

    Parameters
    ----------
    label :
        The label of the `{type.__name__}` object.
    '''

    return get_item


def _create_set_func(cls, name, type):

    def set_item(self, label, item):

        # TODO checked typed items
        if not isinstance(item, type) and \
                not hasattr(type, "_glotaran_model_item_typed"):
            raise TypeError
        getattr(self, name)[label] = item

    set_item.__annotations__ = {
        'label': str,
        'item': type,
    }
    set_item.__name__ = f'set_{name}'
    set_item.__qualname__ = cls.__qualname__ + '.' + set_item.__name__
    set_item.__module__ = cls.__module__
    set_item.__doc__ = f'''
    Sets the `{type.__name__}` object with the given label with to the item.

    Parameters
    ----------
    label :
        The label of the `{type.__name__}` object.
    item :
        The `{type.__name__}` item.
    '''

    return set_item
