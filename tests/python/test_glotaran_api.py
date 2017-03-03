#from glotaran import load

# load('test_glotaran_api_spec.yml')

from glotaran.model import (create_parameter_list, Parameter, InitialConcentration,
                           ZeroConstraint, EqualConstraint, EqualAreaConstraint, FixedConstraint, BoundConstraint,
                           Relation)
from glotaran.models.kinetic import KineticModel, KMatrix, KineticMegacomplex, KineticDatasetDescriptor, KineticSeperableModel
import numpy as np
import matplotlib.pyplot as plt

model = KineticModel()
plist = create_parameter_list([["k1", 0.005], ["k2", 0.001]])
model.parameter = plist
model.add_parameter(Parameter(1, label="j1"))
model.add_parameter(Parameter(0, label="j1"))

model.compartments = ["s1", "s2"]

k_mat = KMatrix("k1", {("s2", "s1"): 1, ("s1", "s2"): 2, ("s2", "s2"): 2})
k_mat = KMatrix("k1", {("s2", "s1"): 1, ("s2", "s2"): 2})

model.add_k_matrix(k_mat)

mc = KineticMegacomplex("mc1", "k1")

model.add_megacomplex(mc)

ic = InitialConcentration("j1", [3, 4])

model.add_initial_concentration(ic)

dset = KineticDatasetDescriptor("d1", "j1", ["mc1"], [], None, None)

model.add_dataset(dset)

print(model)

times = np.asarray(np.arange(0, 1500, 1.5))
test_x = np.arange(12820, 15120, 46)

fitmodel = KineticSeperableModel(model)
data = fitmodel.eval(fitmodel.get_initial_fitting_parameter(), *times, **{'dataset':'d1',
                                           'noise':True, 'noise_std_dev':0.0001,
                                           'd1_x': test_x,
                                           'amplitudes':[10, 20],
                                          'locations':[14700, 13515],
                                           'delta': [50,100]})
#plt.plot(times, data)
fig = plt.figure()
plt.xlabel('Time (ps)')
plt.ylabel('$Wavenumber\ [\ cm^{-1}\ ]$')
plt.pcolormesh(times, test_x, data.T)
plt.show()
# data = fitmodel.eval(fitmodel.get_initial_fitting_parameter(), *times, **{'dataset':'d1', 'd1_x':[0,1,2]})
#plt.plot(times, data[:, 0])
print(data.shape)
