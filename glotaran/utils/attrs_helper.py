"""Helper functions for attrs."""
from glotaran.utils.helpers import nan_or_equal


def no_default_vals_in_repr(cls):
    """Class decorator to omits attributes from repr that have their default value.

    Needs to be on top of the ``attr.define`` decorator.
    Based on: https://stackoverflow.com/a/47663099/3990615

    Parameters
    ----------
    cls
        Class decorated with ``attr.define``.

    Returns
    -------
    type[cls]
    """
    defaults = {
        attribute.name: attribute.default
        for attribute in cls.__attrs_attrs__
        if attribute.repr is True
    }

    def repr_(self) -> str:
        """Return string representing the instance.

        Parameters
        ----------
        self: cls
            Instance of the wrapped class.

        Returns
        -------
        str
        """
        real_cls = self.__class__

        if (qualname := getattr(real_cls, "__qualname__", None)) is not None:
            class_name = qualname.rsplit(">.", 1)[-1]
        else:
            class_name = real_cls.__name__

        args_str = ", ".join(
            f"{name}={repr(getattr(self, name))}"
            for name in defaults
            if not nan_or_equal(getattr(self, name), defaults[name])
        )

        return f"{class_name}({args_str})"

    cls.__repr__ = repr_
    return cls
