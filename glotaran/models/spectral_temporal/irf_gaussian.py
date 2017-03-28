from .irf import Irf


class GaussianIrf(Irf):
    """
    Represents a gaussian IRF.

    One width and one center is a single gauss.

    One center and multiple widths is a multiple gaussian.

    Multiple center and multiple widths is Double-, Triple- , etc. Gaussian.

    Parameter
    ---------

    label: label of the irf
    center: one or more center of the irf as parameter indices
    width: one or more widths of the gaussian as parameter index
    center_dispersion: polynomial coefficients for the dispersion of the
        center as list of parameter indices. None for no dispersion.
    width_dispersion: polynomial coefficients for the dispersion of the
        width as parameter indices. None for no dispersion.

    """
    _scale = None

    def __init__(self, label, center, width, center_dispersion=None,
                 width_dispersion=None, scale=None, normalize=True):
        self.center = center
        self.center_dispersion = center_dispersion
        self.width = width
        self.width_dispersion = width_dispersion
        self.scale = scale
        self.normalize = normalize
        super(GaussianIrf, self).__init__(label)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if not isinstance(value, list):
            raise TypeError("Scale must be list  of parameter indices.")
        if any(not isinstance(val, int) for val in value):
            raise TypeError("Scale must be list  of parameter indices.")
        self._scale = value

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(val, int) for val in value):
            raise TypeError("Parameter indices must be integer.")
        self._center = value

    @property
    def center_dispersion(self):
        return self._center_dispersion

    @center_dispersion.setter
    def center_dispersion(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(val, int) for val in value):
            raise TypeError("Parameter indices must be integer.")
        self._center_dispersion = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(val, int) for val in value):
            raise TypeError("Parameter indices must be integer.")
        self._width = value

    @property
    def width_dispersion(self):
        return self._width_dispersion

    @width_dispersion.setter
    def width_dispersion(self, value):
        if not isinstance(value, list):
            value = [value]
        if any(not isinstance(val, int) for val in value):
            raise TypeError("Parameter indices must be integer.")
        self._width_dispersion = value

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        if not isinstance(value, bool):
            raise TypeError("Normalize must be 'true' or 'false'.")
        self._normalize = value

    def type_string(self):
        t = "'Gaussian'"
        if len(self.center) != len(self.width):
            if len(self.width) is 2:
                t = "'Double Gaussian'"
            elif len(self.width) is 3:
                t = "'Triple Gaussian'"
            elif len(self.width) > 3:
                t = "'{} Gaussian'".format(len(self.width))
        elif len(self.center) is not 1:
            t = "'Multiple Gaussian'"
        return t

    def __str__(self):
        s = """{} Center: {} Width: {} Center Dispersion: {} \
Width Dispersion {} Scale: {}, Nomalize: {}"""
        return s.format(super(GaussianIrf, self).__str__(), self.center,
                        self.width, self.center_dispersion,
                        self.width_dispersion, self.scale, self.normalize)
