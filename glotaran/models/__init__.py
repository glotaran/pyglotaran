"""Glotaran Models Package"""

MODELS = {}


def glotaran_model(name: str):
    """Decorator which registers a model by its name.

    Parameters
    ----------
    name : str
        name is name of the model.

    """
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator
