from .spectral_shape import SpectralShape


class SpectralShapeGaussian(SpectralShape):
    def __init__(self, label, amplitude, location, width):
        super(SpectralShapeGaussian, self).__init__(label)
        self.amplitude = amplitude
        self.location = location
        self.width = width

    def __str__(self):
        return "Label: {} Type: Gaussian\tAmplitude: {}\tLocation: {}\t"\
            "Width: {}"\
            .format(self.label, self.amplitude, self.location, self.width)
