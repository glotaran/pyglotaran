from .decorators import glotaran_model_item

@glotaran_model_item()
class Megacomplex(object):

    def __str__(self):
        return f"### _{self.label}_\n"
