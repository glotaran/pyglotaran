import inspect
from dataclasses import dataclass

class MissingParameterException(Exception):
    pass

@dataclass
class ModelItem():
    _items = []

    label: str

    @classmethod
    def from_dict(cls, item_dict):
        params = inspect.signature(cls.__init__).parameters
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

        return cls(*args)

    @classmethod
    def from_list(cls, item_list):
        params = [p for p in inspect.signature(cls.__init__).parameters]
        if len(item_list) is not len,(params):
            raise MissingParameterException(params[len(item_list)])
        return cls(*item_list)
