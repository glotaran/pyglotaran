from collections import OrderedDict
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from _shared import times_no_irf

from glotaran.model import Parameter
from glotaran.model.parameter_leaf import ParameterLeaf, Parameters
from glotaran.models.spectral_temporal import KineticDatasetDescriptor, KineticMegacomplex, KineticModel, KMatrix
from glotaran.specification_parser import parse_yml

times = times_no_irf()
x = np.asarray([0, 1])

param = Parameter()
param.value = 0.01
param.label = "1"
param.vary = True
sim_pars = ParameterLeaf("p")
sim_pars.add_parameter(param)

eval_pars = Parameters()  # the simulated parameters
eval_pars.add('p_1', 0.01)
eval_pars.pretty_print()

sim_model = KineticModel()
sim_model.parameter = sim_pars
sim_model.compartments = ['s1']
sim_model.add_megacomplex(KineticMegacomplex('mc1', ['kmat1']))
sim_model.add_dataset(KineticDatasetDescriptor('dataset1', None,['mc1'], {}, None, {}, None, {}))
kmat1 = OrderedDict({("s1", "s1"): 1})
sim_model.add_k_matrix(KMatrix("kmat1", kmat1,{"s1"}))
sim_model.eval('dataset1',  {"spectral": x, "time": times, }, parameter=eval_pars)
sim_data = sim_model.datasets['dataset1'].data.data

plt.xlabel('Time (ps)')
plt.ylabel('Intensity')
print("len(times): {}".format(len(times)))
print("sim_data.shape: {}".format(sim_data.shape))

plt.plot(times, sim_data[1,:], label="680nm")
plt.legend(borderaxespad=1.)
plt.show(block=False)

#  TODO: Joern why is this failing?
# print(sim_model)

print(sim_model.parameter.as_parameters_dict())
fit_model = copy(sim_model)
fit_model.parameter.get("1").value = 0.05
result = fit_model.fit()
result.best_fit_parameter.pretty_print()

#  TODO: Joern why is this failing?
# print(fit_model)

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
#  TODO: Joern why is this failing?
print(specfit_model)
specfit_result = specfit_model.fit()
specfit_result.best_fit_parameter.pretty_print()

result_data = specfit_result.eval(*times, **{'dataset1': sim_data})

plt.plot(times, sim_data, label="data")
plt.plot(times, result_data, label="fit")

plt.show()
