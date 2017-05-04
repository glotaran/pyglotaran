from collections import OrderedDict
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from _shared import times_no_irf

from glotaran.model import Parameter
from glotaran.model.parameter_leaf import ParameterLeaf, Parameters
from glotaran.models.spectral_temporal import KineticDatasetDescriptor, KineticMegacomplex, KineticModel, KMatrix
from glotaran.specification_parser import parse_yml

# Initializing the (common) times and spectral_indices vectors:
times = times_no_irf()
spectral_indices = np.asarray([0, 1])

# Defining the parameter(s) for which the model will be initially evaluated
# to produce the simulated data
eval_pars = Parameters()  # the simulated parameters
eval_pars.add('p_1', 0.0123456789)
eval_pars.pretty_print()

############################################################################
## Using the API to define the model specification to fit the simulated data

# Define a Parameter and add it to parameter collection "model_spec_pars"
# which is an OrderedDict
param = Parameter()
param.value = 0.05
param.label = "1"
param.vary = True
model_spec_pars = ParameterLeaf("p")
model_spec_pars.add_parameter(param)

# Initialize the model which is a
# KineticModel class defined in glotaran.models.spectral_temporal.model
# derived from the Model class defined in glotaran.model.model.py
sim_model = KineticModel()
# Set the parameter attribute to the earlier defined collection of parameters
sim_model.parameter = model_spec_pars
# Set the compartments attribute
sim_model.compartments = ['s1']
# Add a megacomplex definition
sim_model.add_megacomplex(KineticMegacomplex('mc1', ['kmat1']))


# Define the kmatrix referenced in the megacomplex
kmat1 = OrderedDict({("s1", "s1"): 1})
sim_model.add_k_matrix(KMatrix("kmat1", kmat1,{"s1"}))
# Add a KineticDatasetDescriptor descriptor
#  label, initial_concentration, megacomplexes, megacomplex_scalings,
#  dataset_scaling, compartment_scalings, irf, shapes
kinDatasetDescriptor = KineticDatasetDescriptor('dataset1', None,  # label, initial_concentration
                                                ['mc1'], {},   # megacomplexes, megacomplex_scalings,
                                                None, {},  # dataset_scaling, compartment_scalings
                                                None, {})  # irf, shapes
sim_model.add_dataset(kinDatasetDescriptor)
# sim_model.eval('dataset1',  {"spectral": spectral_indices, "time": times })
sim_model.eval('dataset1',  {"spectral": spectral_indices, "time": times, }, parameter=eval_pars)
sim_data = sim_model.datasets['dataset1'].data.data

plt.xlabel('Time (ps)')
plt.ylabel('Intensity')
print("len(times): {}".format(len(times)))
print("sim_data.shape: {}".format(sim_data.shape))

plt.plot(times, sim_data[1,:], label="680nm")
plt.legend(borderaxespad=1.)
plt.show(block=False)
#  TODO: fix print command for Model
# print(sim_model)

print(sim_model.parameter.as_parameters_dict())
fit_model = copy(sim_model)
fit_model.parameter.get("1").value = 0.05
result = fit_model.fit()
result.best_fit_parameter.pretty_print()

# print(fit_model)

############################################################################
## Using the API to define the model specification to fit the simulated data

fitspec = '''
type: kinetic

parameters:
 - 0.05

compartments: [s1]

megacomplexes:
- label: mc1
  k_matrices: [kmat1]

k_matrices:
  - label: "kmat1"
    matrix: {
      '("s1","s1")': 1
    }

datasets:
- label: dataset1
  type: spectral
  megacomplexes: [mc1]
  path: ''

'''

specfit_model = parse_yml(fitspec)
specfit_model.eval('dataset1',  {"spectral": spectral_indices, "time": times},  parameter=eval_pars)

# print(specfit_model)
specfit_result = specfit_model.fit()
specfit_result.best_fit_parameter.pretty_print()
# TODO: implement return method for Result object besides best_fit_parameter
# also estimated spectra (i.e. EAS, DAS, SAS depending on model), reconstructed DAS (EAS?), SVD, residuals
print(specfit_result.svd())

# Statement to get simulated dataset back based on best_fit_parameters
result_data = specfit_result.eval('dataset1')

plt.plot(times, sim_data, label="data")
plt.plot(times, result_data, label="fit")

plt.show()
