import time

import numpy as np
from _shared import times_with_irf

from glotaran.model.parameter_leaf import Parameters
from glotaran.specification_parser import parse_yml

# Initializing the (common) times and spectral_indices vectors:
times1 = times_with_irf()
times2 = np.concatenate((times_with_irf(),np.arange(3100, 3500, 100)))
spectral_indices1 = np.asarray([1, 2])
spectral_indices2 = np.asarray([2, 4])

# Defining the parameter(s) for which the model will be initially evaluated
# to produce the simulated data
eval_pars = Parameters()  # the simulated parameters
eval_pars.add('p_1', 0.05123)
eval_pars.add('p_2', 0.00101)
eval_pars.add('p_3', 0.04876)
eval_pars.add('p_4', -0.99999)
eval_pars.add('p_5', 0.01)
eval_pars.pretty_print()

# print(fit_model)

############################################################################
## Using the API to define the model specification to fit the simulated data

fitspec = '''
type: kinetic

parameters:
 - 0.055
 - 0.01
 - 0.035
 - -2.3
 - 0.005

compartments: [s1, s2, s3]

megacomplexes:
- label: mc1
  k_matrices: [k1]
- label: mc2
  k_matrices: [k2]

k_matrices:
  - label: "k1"
    matrix: {
      '("s2","s1")': 1,
      '("s2","s2")': 2,
    }
  - label: "k2"
    matrix: {
      '("s2","s1")': 1,
      '("s2","s2")': 2,
      '("s3","s3")': 3,
    }

irf:
  - label: irf1
    type: gaussian
    center: 4
    width: 5

datasets:
- label: dataset1
  type: spectral
  megacomplexes: [mc1]
  path: ''
  irf: irf1
- label: dataset2
  type: spectral
  megacomplexes: [mc2]
  path: ''
  irf: irf1

'''

specfit_model = parse_yml(fitspec)
print(specfit_model,flush=True)
time.sleep(1)
print('times1.shape = {} and times1[0] = {}'.format(times1.shape, times1[0]))
print('times2.shape = {} and times2[0] = {}'.format(times2.shape, times2[0]))
time.sleep(1)
specfit_model.eval('dataset1', {"spectral": spectral_indices1, "time": times1})
specfit_model.eval('dataset2', {"spectral": spectral_indices2, "time": times2})

specfit_result = specfit_model.fit()
specfit_result.best_fit_parameter.pretty_print()

