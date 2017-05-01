from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from _shared import times_no_irf
from lmfit import Parameters

from glotaran.models.spectral_temporal import KineticDatasetDescriptor, KineticMegacomplex, KineticModel, KMatrix
from glotaran.specification_parser import parse_yml

times = times_no_irf()
x = np.asarray([0, 1])

sim_pars = Parameters()  # the simulated parameters
sim_pars.add('p_1', 0.01)
sim_pars.pretty_print()

sim_model = KineticModel()
sim_model.compartments = ['s1']
sim_model.add_megacomplex(KineticMegacomplex('mc1', ['kmat1']))
sim_model.add_dataset(KineticDatasetDescriptor('dataset1', None,['mc1'], {}, None, {}, None, {}))
kmat1 = OrderedDict()
kmat1['s1,s1'] = 1
sim_model.add_k_matrix(KMatrix("kmat1", kmat1,"s1"))
sim_model.eval('dataset1',  {"time": times, "spectral": x}, parameter=sim_pars)

print(sim_model)

sim_model.compartments = ["s1"]
sim_model.add_k_matrix(KMatrix("k1", {("s1", "s1"): 1}))
sim_model.add_megacomplex(KineticMegacomplex("mc1", "k1"))
sim_model.add_initial_concentration(InitialConcentration("j1", [2]))
sim_model.add_dataset(("d1", "j1", ["mc1"], [], None, None))
print(sim_model)

times = times_no_irf()
test_x = np.array([680])
kin_sim_model = KineticSeparableModel(sim_model)
sim_data = kin_sim_model.eval(kin_sim_model.get_initial_fitting_parameter(), *times, **{'dataset':'d1', 'noise': True, 'noise_std_dev': 0.1, 'noise_seed': 1.5, 'd1_x':test_x})

plt.xlabel('Time (ps)')
plt.ylabel('Intensity')
plt.plot(times, sim_data, label="680nm")
plt.legend(borderaxespad=1.)
plt.show()

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

initial_concentrations: []

irf: []

datasets:
- label: dataset1
  type: spectral
  megacomplexes: [mc1]
  path: ''

'''

fit_model = parse_yml(fitspec)
fit_result = fit_model.fit()
fit_result.best_fit_parameter.pretty_print()

result_data = fit_result.eval(*times, **{'dataset1': sim_data})

plt.plot(times, sim_data, label="data")
plt.plot(times, result_data, label="fit")

plt.show()
