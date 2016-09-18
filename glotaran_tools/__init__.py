from glotaran_tools import (timetrace, model_spec_yaml,
                            spectral_timetrace, simulated_spectral_timetrace,
                            wavelength_time_explicit_file)

Timetrace = timetrace.Timetrace
SpectralTimetrace = spectral_timetrace.SpectralTimetrace
SimulatedSpectralTimetrace = simulated_spectral_timetrace\
    .SimulatedSpectralTimetrace
WavelengthExplicitFile = wavelength_time_explicit_file.WavelengthExplicitFile
TimeExplicitFile = wavelength_time_explicit_file.TimeExplicitFile

parse_file = model_spec_yaml.parse_file
