from _shared import times_no_irf
from _shared import times_with_irf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lmfit import Parameters
from glotaran.specification_parser import parse_yml
from glotaran.models.kinetic import KineticSeparableModel
from glotaran.model import (create_parameter_list, Parameter, InitialConcentration,
                            ZeroConstraint, EqualConstraint, EqualAreaConstraint, FixedConstraint, BoundConstraint,
                            Relation)
from glotaran.models.kinetic import KineticModel, KMatrix, KineticMegacomplex, KineticDatasetDescriptor, \
    KineticSeparableModel

sim_model = KineticModel()
sim_model.parameter = create_parameter_list([["k1", 0.01], 1])
sim_model.compartments = ["s1"]
sim_model.add_k_matrix(KMatrix("k1", {("s1", "s1"): 1}))
sim_model.add_megacomplex(KineticMegacomplex("mc1", "k1"))
sim_model.add_initial_concentration(InitialConcentration("j1", [2]))
sim_model.add_dataset(KineticDatasetDescriptor("d1", "j1", ["mc1"], [], None, None))
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
  k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {
      '("s1","s1")': 1
    }

initial_concentrations: []

irf: []

io:
- label: dataset1
  type: spectral
  megacomplexes: [mc1]
  path: ''

'''

fit_model = parse_yml(fitspec)
kin_fit_model = KineticSeparableModel(fit_model)
print(sim_data)
fit_result = kin_fit_model.fit(kin_fit_model.get_initial_fitting_parameter(),
                               *times, **{"dataset1": sim_data})
print(sim_data)
fit_result.best_fit_parameter.pretty_print()
result_data = fit_result.eval(*times, **{'dataset1': sim_data})

plt.plot(times, sim_data, label="data")
plt.plot(times, result_data, label="fit")

plt.show()
