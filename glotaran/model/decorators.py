import inspect
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict


class MissingParameterException(Exception):
    pass


def glotaran_model(name, attributes={}):

    def decorator(cls):

        setattr(cls, 'model_type', name)
        print(attributes)
        setattr(cls, '__annotations__', {})

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

        print(getattr(cls, '__annotations__'))
        setattr(cls, '__init__', init)

        return cls

    return decorator

def glotaran_model_attribute(name):
    def decorator(attr):
        attr.gta_attr = name
        attr.gta_attr_add = attr.__name__
        print(name, attr.__name__)
        return attr
    return decorator

def glotaran_model_item(attributes={}, parameter=[]):
    def decorator(cls):

        # Set annotations for autocomplete and doc

        annotations = {'label': str}
        for name, atype in attributes.items():
            annotations[name] = atype
        setattr(cls, '__annotations__', annotations)

        # We turn it into dataclass to get automagic inits
        cls = dataclass(cls)

        # store the fit parameter for later sanity checking
        setattr(cls, '_glotaran_fit_parameter', parameter)

        # strore the number of attributes
        setattr(cls, '_glotaran_nr_attributes', len(attributes))

        # now we want nice class methods for serializing

        @classmethod
        def from_dict(ncls, item_dict):
            params = inspect.signature(ncls.__init__).parameters
            args = []
            for name, param in params.items():
                if name == "self":
                    continue
                if name not in item_dict:
                    if param.default is param.empty:
                        raise MissingParameterException(name)
                    args.append(param.default)
                else:
                    args.append(item_dict[name])

            return ncls(*args)

        setattr(cls, 'from_dict', from_dict)

        @classmethod
        def from_list(ncls, item_list):
            params = [p for p in inspect.signature(ncls.__init__).parameters]
            # params contains 'self'
            if len(item_list) is not len(params)-1:
                raise MissingParameterException
            print(item_list)

            return ncls(*item_list)


        setattr(cls, 'from_list', from_list)
        return cls

    return decorator
