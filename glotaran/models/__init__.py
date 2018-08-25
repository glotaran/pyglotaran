"""Glotaran Models Package"""

MODELS = {}


def glotaran_model(cls):
    MODELS[cls.type_string()] = cls
    return cls
