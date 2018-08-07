from .spectral_shape import SpectralShape


class SpectralShapeGaussian(SpectralShape):
    def __init__(self, label, amplitude, location, width):
        super(SpectralShapeGaussian, self).__init__(label)
        self.amplitude = amplitude
        self.location = location
        self.width = width

    def __str__(self):
        string = super(SpectralShapeGaussian, self).__str__()
        string += ", _Type_: Gaussian"
        string += f", _Amplitude_: {self.amplitude}"
        string += f", _Location_: {self.location}"
        string += f", _Width_: {self.width}"
        return string
