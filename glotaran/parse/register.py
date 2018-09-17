"""A register for models"""


from glotaran.model.base_model import BaseModel

_model_register = {}


def register_model(model_type: str, model: BaseModel):
    """register_model registers a model.

    Parameters
    ----------
    model_type : str
        model_type is type of the model.
    model : glotaran.model.base_model
        model is the model to be registered.
    """
    _model_register[model_type] = model


def known_model(model_type: str) -> bool:
    """known_model returns True if the model_type is in the register.

    Parameters
    ----------
    model_type : str
        model_type is type of the model.

    Returns
    -------
    known : bool
    """
    return model_type in _model_register


def get_model(model_type: str) -> BaseModel:
    """get_model gets a model from the register.

    Parameters
    ----------
    model_type : str
        model_type is type of the model.

    Returns
    -------
    model : glotaran.model.base_model
    """
    return _model_register[model_type]
