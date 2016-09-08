from .spectral_timetrace import SpectralTimetrace
import math as m


def square(x):
    return x*x


class SimulatedSpectralTimetrace(SpectralTimetrace):

    def __init__(self, amplitudes, rates, positions, widths, spectrum_min,
                 spectrum_max, spectrum_delta, nr_timepoints,
                 delta_timepoints, label="simulation"):
        if any(position > spectrum_max * spectrum_delta or
               position < spectrum_min for position in positions):
            raise Exception("Positions must be in spectral range.")

        if not len(rates) == len(positions) or not len(rates) == len(widths):
            raise Exception("""Number of amplitudes, rates, positions and widths
                            must be equal""")

        timepoints = []
        for i in range(0, nr_timepoints):
            timepoints.append(i*delta_timepoints)

        spectral_indices = []
        for i in range((spectrum_max-spectrum_min)//spectrum_delta):
            spectral_indices.append(spectrum_min+i*spectrum_max)
        channels = []
        for i in range(len(spectral_indices)):
            channels.append([])
            for j in range(len(timepoints)):
                val = 0
                for k in range(len(rates)):
                    val += amplitudes[k] * \
                            m.exp(-m.log(2) *
                                  square(2*(spectral_indices[i] -
                                            positions[k]) /
                                         widths[k])) * \
                            m.exp(timepoints[j]*rates[k])
                channels[i].append(val)
        super(SimulatedSpectralTimetrace, self).__init__(label, channels,
                                                         timepoints,
                                                         spectral_indices)
