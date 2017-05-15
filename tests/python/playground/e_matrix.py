# TODO: Playground File
import numpy as np
from lmfit import Parameters

from glotaran.specification_parser import parse_yml

fitspec = '''
type: kinetic

parameters:
    - 10e-3
    - 50e-4

compartments: [s1, s2]

megacomplexes:
- label: mc1
  k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {{
      '("s1","s1")': 1,
      '("s2","s2")': 2
    }}

initial_concentrations: []

datasets:
- label: dataset1
  type: spectral
  megacomplexes: [mc1]
  path: ''
'''

initial_parameter = [101e-4]
initial_parameter = [505e-5]

times = np.arange(0, 500, 1.5)

wanted_params = Parameters()
wanted_params.add("p_1", 101e-3)
wanted_params.add("p_2", 505e-4)

fitmodel = parse_yml(fitspec.format(initial_parameter))
x = np.arange(12820, 15120, 46)


fitmodel.eval('dataset1',  {"time": times, "spectral": x},
              parameter=wanted_params)

print(fitmodel.datasets['dataset1'].data.data.shape)

cmat = np.asarray(fitmodel.c_matrix())
emat = np.asarray(fitmodel.e_matrix())

print(cmat.shape)
print(emat.shape)

result = fitmodel.fit()

print(np.asarray(result.c_matrix(**{'dataset': 'dataset1'})).shape)
print(result.final_residual().shape)
print(np.asarray(result.e_matrix("dataset1")).shape)
