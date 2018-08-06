from .irf import Irf


class GaussianIrf(Irf):
    """
    Represents a gaussian IRF.

    One width and one center is a single gauss.

    One center and multiple widths is a multiple gaussian.

    Multiple center and multiple widths is Double-, Triple- , etc. Gaussian.

    Parameters
    ----------

    label:
        label of the irf
    center:
        one or more center of the irf as parameter indices
    width:
        one or more widths of the gaussian as parameter index
    center_dispersion:
        polynomial coefficients for the dispersion of the
        center as list of parameter indices. None for no dispersion.
    width_dispersion:
        polynomial coefficients for the dispersion of the
        width as parameter indices. None for no dispersion.

    """
    _scale = None

    def __init__(self, label, center, width, center_dispersion=None,
                 width_dispersion=None, scale=None, normalize=True,
                 backsweep=False, backsweep_period=None):
        self.center = center
        self.center_dispersion = center_dispersion
        self.width = width
        self.width_dispersion = width_dispersion
        self.scale = scale
        self.normalize = normalize
        self.backsweep = backsweep
        self.backsweep_period = backsweep_period
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
        self._center = value

    @property
    def center_dispersion(self):
        return self._center_dispersion

    @center_dispersion.setter
    def center_dispersion(self, value):
        if not isinstance(value, list):
            value = [value]
        self._center_dispersion = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if not isinstance(value, list):
            value = [value]
        self._width = value

    @property
    def width_dispersion(self):
        return self._width_dispersion

    @width_dispersion.setter
    def width_dispersion(self, value):
        if not isinstance(value, list):
            value = [value]
        self._width_dispersion = value

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        if not isinstance(value, bool):
            raise TypeError("Normalize must be 'true' or 'false'.")
        self._normalize = value

    @property
    def backsweep(self):
        """True or false """
        return self._backsweep

    @backsweep.setter
    def backsweep(self, value):
        """

        Parameters
        ----------
        value : True or False


        Returns
        -------


        """
        if not isinstance(value, bool):
            raise TypeError("Backsweep must be True or False")
        self._backsweep = value

    @property
    def backsweep_period(self):
        """Parameter Index"""
        return self._backsweep_period

    @backsweep_period.setter
    def backsweep_period(self, value):
        """

        Parameters
        ----------
        value : Parameter Index


        Returns
        -------


        """
        self._backsweep_period = value

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
        string = super(GaussianIrf, self).__str__()
        string += f"* _Center_: {self.center}\n"
        string += f"* _Width_: {self.width}\n"
        string += f"* _Center Dispersion_: {self.center_dispersion}\n"
        string += f"* _Width Dispersion_: {self.width_dispersion}\n"
        string += f"* _Scale_: {self.scale}\n"
        string += f"* _Nomalize_: {self.normalize}\n"
        return string
