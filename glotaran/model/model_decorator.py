import typing

from glotaran.parse.register import register_model

from .dataset_descriptor import DatasetDescriptor
from .util import wrap_func_as_method


def model(name,
          attributes={},
          dataset_type=DatasetDescriptor,
          megacomplex_type=None,
          calculated_matrix=None,
          estimated_matrix=None,
          calculated_axis=None,
          estimated_axis=None,
          constrain_calculated_matrix_function=None,
          additional_penalty_function=None,
          finalize_result_function=None,
          allow_grouping=True,
          ):

    def decorator(cls):

        setattr(cls, '_model_type', name)
        setattr(cls, '_finalize_result', finalize_result_function)
        setattr(cls, '_constrain_calculated_matrix_function',
                constrain_calculated_matrix_function)
        setattr(cls, '_additional_penalty_function',
                additional_penalty_function)
        setattr(cls, '_allow_grouping', allow_grouping)

        if calculated_matrix:
            c_mat = wrap_func_as_method(cls, name='calculated_matrix')(calculated_matrix)
            setattr(cls, 'calculated_matrix', c_mat)
        setattr(cls, 'calculated_axis', calculated_axis)

        if estimated_matrix:
            e_mat = wrap_func_as_method(cls, name='estimated_matrix')(estimated_matrix)
            setattr(cls, 'estimated_matrix', e_mat)
        setattr(cls, 'estimated_axis', estimated_axis)

        if not hasattr(cls, '_glotaran_model_attributes'):
            setattr(cls, '_glotaran_model_attributes', {})
        else:
            setattr(cls, '_glotaran_model_attributes',
                    getattr(cls, '_glotaran_model_attributes').copy())

        attributes['megacomplex'] = megacomplex_type
        attributes['dataset'] = dataset_type

        # Set annotations and methods for attributes

        for attr_name, attr_type in attributes.items():
            getattr(cls, '_glotaran_model_attributes')[attr_name] = None

            attr_prop = _create_property_for_attribute(cls, attr_name, attr_type)
            setattr(cls, attr_name, attr_prop)

            if getattr(attr_type, '_glotaran_has_label'):
                get_item = _create_get_func(cls, attr_name, attr_type)
                setattr(cls, get_item.__name__, get_item)
                set_item = _create_set_func(cls, attr_name, attr_type)
                setattr(cls, set_item.__name__, set_item)

            else:
                add_item = _create_add_func(cls, attr_name, attr_type)
                setattr(cls, add_item.__name__, add_item)
                setattr(cls, attr_name, [])

        init = _create_init_func(cls, attributes)
        setattr(cls, '__init__', init)

        register_model(name, cls)

        return cls

    return decorator


def _create_init_func(cls, attributes):

    @wrap_func_as_method(cls)
    def __init__(self):
        for attr_name, attr_item in attributes.items():
            if getattr(attr_item, '_glotaran_has_label'):
                setattr(self, f'_{attr_name}', {})
            else:
                setattr(self, f'_{attr_name}', [])
        super(cls, self).__init__()

    return __init__


def _create_add_func(cls, name, type):

    @wrap_func_as_method(cls, name=f'add_{name}')
    def add_item(self, item: type):
        f'''Adds an `{type.__name__}` object.

        Parameters
        ----------
        item :
            The `{type.__name__}` item.
        '''

        if not isinstance(item, type):
            if not hasattr(type, "_glotaran_model_item_typed") or \
               not isinstance(item, tuple(getattr(type, '_glotaran_model_item_types').values())):
                raise TypeError
        getattr(self, f'_{name}').append(item)

    return add_item


def _create_get_func(cls, name, type):

    @wrap_func_as_method(cls, name=f'get_{name}')
    def get_item(self, label: str) -> type:
        f'''
        Returns the `{type.__name__}` object with the given label.

        Parameters
        ----------
        label :
            The label of the `{type.__name__}` object.
        '''
        return getattr(self, f'_{name}')[label]

    return get_item


def _create_set_func(cls, name, type):

    @wrap_func_as_method(cls, name=f'set_{name}')
    def set_item(self, label: str, item: type):
        f'''
        Sets the `{type.__name__}` object with the given label with to the item.

        Parameters
        ----------
        label :
            The label of the `{type.__name__}` object.
        item :
            The `{type.__name__}` item.
        '''

        if not isinstance(item, type):
            if not hasattr(type, "_glotaran_model_item_typed") or \
               not isinstance(item, tuple(getattr(type, '_glotaran_model_item_types').values())):
                raise TypeError
        getattr(self, f'_{name}')[label] = item

    return set_item


def _create_property_for_attribute(cls, name, type):

    return_type = typing.Dict[str, type] if hasattr(type, '_glotaran_has_label') \
            else typing.List[type]

    doc_type = 'dictonary' if hasattr(type, '_glotaran_has_label') else 'list'

    @property
    @wrap_func_as_method(cls, name=f'{name}')
    def attribute(self) -> return_type:
        f'''A {doc_type} containing {type.__name__}'''
        return getattr(self, f'_{name}')

    return attribute
