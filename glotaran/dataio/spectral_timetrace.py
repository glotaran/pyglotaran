from enum import Enum


class SpectralUnit(Enum):
    pixel = 0
    nm = 1
    per_cm = 2


class SpectralTimetrace(object):
    """
    Represents a spectral timetrace
    """
    def __init__(self, spectra, timepoints, spectral_indices=[],
                 spectral_unit=SpectralUnit.pixel, timeunit="s"):
        if not isinstance(spectral_unit, SpectralUnit):
            raise TypeError
        self.spectral_unit = spectral_unit
        if spectral_indices is []:
            for i in range(spectra[0]):
                spectral_indices.append(i)
        self._spectral_indices = spectral_indices
        channel_labels = []
        for i in spectral_indices:
            channel_labels.append(i)

    def wavenumbers(self):
        if self.spectral_unit is SpectralUnit.nm:
            wn = []
            for wl in self.wavelengths():
                wn.append(10000000/wl)
            return wn
        elif self.spectral_unit is SpectralUnit.per_cm:
            return self.spectral_indices
        else:
            raise Exception("Spectral unit is pixel.")

    def wavelengths(self):
        if self.spectral_unit is SpectralUnit.per_cm:
            wl = []
            for wn in self.wavenumbers():
                wl.append(10000000/wn)
            return wl
        elif self.spectral_unit is SpectralUnit.nm:
            return self.spectral_indices
        else:
            raise Exception("Spectral unit is pixel.")
