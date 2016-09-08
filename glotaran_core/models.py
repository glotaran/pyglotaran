try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
from .model import Model


class Models(OrderedDict):
    """
    A dictionary of all models to be fitted.
    """
    def __init__(self, models):
        super(Models, self)
        if not isinstance(models, list):
            models = [models]
        if any(not isinstance(model, Model) for model in models):
            raise TypeError

        for model in models:
            self.add(model)

    def add(self, model):
        """
        Add a model.
        """
        if not isinstance(model, Model):
            raise TypeError

        if len(model.labels()) != set(model.labels()) or any(label in self for
                                                             label in
                                                             model.labels()):
            raise Exception("Labels must be unique")

        for label in model.labels:
            self.__setitem__(label, model)
