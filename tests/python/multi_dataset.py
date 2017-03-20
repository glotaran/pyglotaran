
import numpy as np

from lmfit import Parameters

from glotaran.specification_parser import parse_yml
from glotaran.model import Dataset, IndependentAxies
from glotaran.models.kinetic import KineticSeparableModel
from glotaran.models.kinetic.c_matrix_generator import CMatrixGenerator

fitspec = '''
type: kinetic

parameters: {}

compartments: [s1, s2]

megacomplexes:
- label: mc1
  k_matrices: [k1]
- label: mc2
  k_matrices: [k2]

k_matrices:
  - label: "k1"
    matrix: {{
      '("s1","s2")': 1,
      '("s2","s2")': 2,
    }}
  - label: "k2"
    matrix: {{
      '("s3","s3")': 3,
    }}

initial_concentrations: []

irf:
  - label: irf1
    type: gaussian
    center: 2
    width: 3


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

initial_parameter = [101e-4, 0, 5]
times = np.asarray(np.arange(-100, 1500, 1.5))
# x = np.arange(12820, 15120, 4.6)
x1 = np.asarray([1.0, 2.0, 3.0])
x2 = np.asarray([2.0, 3.1, 4.0])

axies1 = IndependentAxies()
axies1.add(x1)
axies1.add(times)

axies2 = IndependentAxies()
axies2.add(x2)
axies2.add(times)

wanted_params = Parameters()
wanted_params.add("p1", 101e-3)
wanted_params.add("p2", 0.3)
wanted_params.add("p3", 10)

model = parse_yml(fitspec.format(initial_parameter))


model.eval(wanted_params, 'dataset1', axies1)
model.eval(wanted_params, 'dataset2', axies2)

gen = CMatrixGenerator(model)

for g in gen.groups():
    print(g.id)
    for c in g.c_matrices:
        print(c.x)
        print(c._dataset)
        print(c.compartment_order)

c_mat = gen.calculate(wanted_params)

for c in c_mat:
    print(c.shape)
#  fitmodel = KineticSeparableModel(model)

#print(fitmodel._get_construction_order())
#  print(fitmodel.c_matrix(initial_parameter))


# print(model.datasets['dataset1'].data.data.shape)


def fit():
    fitmodel.fit(fitmodel.get_initial_fitting_parameter()).best_fit_parameter.pretty_print()
    pass

if __name__ == '__main__':
    #  fit()
    pass
