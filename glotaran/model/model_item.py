from dataclasses import dataclass
import inspect


class MissingParameterException(Exception):
    pass


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
