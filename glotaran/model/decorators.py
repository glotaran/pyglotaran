from dataclasses import dataclass

# Decorators

def glotaran_model(cls):
    print(cls)


    def set_item(self, label, item):
        if not isinstance(item, cls):
            raise Exception
        getattr(self, f"{name}")[label] = item

    for name in cls.__dict__:
        method = cls.__dict__[name]
        if hasattr(method, "gta_attr"):
            # do something with the method and class
            attr = getattr(method, "gta_attr")
            attr_add = getattr(method, "gta_attr_add")
            print(name, cls)
            cls._attributes[attr] = attr_add
    return cls


def glotaran_attribute(name):
    def decorator(attr):
        attr.gta_attr = name
        attr.gta_attr_add = attr.__name__
        print(name, attr.__name__)
        return attr
    return decorator

def glotaran_model_item(parameter=[]):
    def decorator(cls):
        cls = dataclass(cls)
        cls._gta_parameter = parameter
        return cls
    return decorator
