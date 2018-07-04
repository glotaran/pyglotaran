# Glotaran package __init__.py

from glotaran.specification_parser import parse_yml
from glotaran.specification_parser import parse_file

__version__ = '0.0.1'

# Top level API
# SeparableModel = separable_model.SeparableModel
# SeparableModelResult = result.SeparableModelResult

# loadSomethingFromFile
# Load Glotaran analysis protocol (*.gat or *.yml) from file

load = parse_file  # short for loadProtocol
parse = parse_yml  # short for loadProtocol

# # user calls this object whatever
# glotaran._loadProtocol()
# glotaran._loadProtocolFromFile() #or path
# glotaran._loadProtocolFromString()
#
# # data file vs dataset ??
# glotaran.add_dataset()
# glotaran._add_dataset_from_file() #or path
# glotaran._add_dataset_from_clipboard()
#
# glotaran.read_data_file
# glotaran.read_multiple_data_files
#
# # model specification
#
# glotaran.free_parameter('parameter')
# glotaran.free_parameter_for_dataset(parameter, index)
#
# ## model evaluation
#
# glotaran.fit() #use lmfit to minimize
# glotaran.calculate() #evaluate the current model at given timepoints and wavelengths and produce data object
#
#
# ## getting results
#
# glotaran.parameters
# glotaran.get_fitted_parameters
#
# glotaran.das
# glotaran.get_decay_associated_spectra
#
# glotaran.eas
# glotaran.get_evolution_associated_spectra
#
# glotaran.sas
# glotaran.get_species_associated_spectra
#
# glotaran.concentrations
# glotaran.get_concentrations
#
# glotaran.residuals
# glotaran.get_residuals
# glotaran.get_residuals_for_dataset(index i)
#
# glotaran.plot_das
# glotaran.plot_eas
# glotaran.plot_sas
# glotaran.plot_concentrations
# glotaran.plot_spectra
