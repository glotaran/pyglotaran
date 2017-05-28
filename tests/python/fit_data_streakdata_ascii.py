import matplotlib.pyplot as plt
import numpy as np
import os

from glotaran.io.wavelength_time_explicit_file import ExplicitFile
from glotaran.specification_parser import parse_yml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_path = os.path.join(THIS_DIR, '..', 'resources', 'data', 'streakdata.ascii')

# Read in streakdata.ascii from resources/data sub-folder
data_file_te = ExplicitFile(root_data_path)
dataset_te = data_file_te.read("dataset1")
times = dataset_te.get_axis("time")
times_shifted = list(np.asarray(times) + 83)
wavelengths = dataset_te.get_axis("spectral")

# # Get data limits
# if reproduce_figures_from_paper:
#     [xmin, xmax] = [-20, 200] #with respect to maximum of IRF (needs function written)
#     [ymin, ymax] = [630,770]
#     linear_range = [-20, 20]
# else:
#     [xmin,xmax] = [min(dataset_te.get_axis("time")), max(dataset_te.get_axis("time"))]
#     [ymin, ymax] = [min(dataset_te.get_axis("spec")),max(dataset_te.get_axis("spec"))]
#     linear_range = [-20, 20]
# print([xmin,xmax,ymin,ymax])

plt.figure()
plt.pcolormesh(times, wavelengths, dataset_te.data)
plt.show(block=False)

fitspec = '''
type: kinetic

parameters: 
 - -83.0
 - 1.5
 - 0.2
 - 0.02
 - 0.07
 - 0.00016
 - {}

irf:
  - label: irf
    type: gaussian
    center: 1
    width: 2
    backsweep: True
    backsweep_period: {}

compartments: [s1, s2, s3, s4]

megacomplexes:
    - label: mc1
      k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {{
      '("s1","s1")': 3,
      '("s2","s2")': 4,
      '("s3","s3")': 5,
      '("s4","s4")': 6
    }}

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''
    irf: irf

'''

# only the last 2 test strings work
defaultTestCase = ("13200.0", "7")
testCases = [("[13200.0, false]", "7"),
             ("[13200.0, {vary: false}]", "7"),
             ("[13200.0, {fit: false}]", "7"),
             ("[13200.0, \"backsweep_period\", {vary: true}]", "backsweep_period"),
             ("[13200.0, true]", "7")
             ]

for spec in testCases:
    specfit_model = parse_yml(fitspec.format(*spec))
    # TODO: fix printing model
    # print(specfit_model)
    times = np.asarray(dataset_te.get_axis("time"))
    wavelengths = np.asarray(dataset_te.get_axis("spectral"))
    specfit_model.datasets['dataset1'].data = dataset_te
    specfit_result = specfit_model.fit()
    specfit_result.best_fit_parameter.pretty_print()

specfit_model = parse_yml(fitspec.format(*defaultTestCase))
times = np.asarray(dataset_te.get_axis("time"))
wavelengths = np.asarray(dataset_te.get_axis("spectral"))
specfit_model.datasets['dataset1'].data = dataset_te
specfit_result = specfit_model.fit()
specfit_result.best_fit_parameter.pretty_print()
residual = specfit_result.final_residual()

levels = np.linspace(0, max(dataset_te.data.flatten()), 10)
cnt = plt.contourf(times, wavelengths, residual, levels=levels, cmap="Greys")
# This is the fix for the white lines between contour levels
for c in cnt.collections:
    c.set_edgecolor("face")
plt.show(block=False)

residual_svd = specfit_result.final_residual_svd()
# Plot left singular vectors (LSV, times, first 3)
plt.subplot(3, 3, 4)
plt.title('LSV')
for i in range(3):
    plt.plot(times,residual_svd[2][i, :])
# Plot singular values (SV)
plt.subplot(3, 3, 5)
plt.plot(range(min(len(times),len(wavelengths))),residual_svd[1],'ro')
plt.yscale('log')
# Plot right singular vectors (RSV, wavelengths, first 3)
plt.subplot(3, 3, 6)
plt.title('RSV')
for i in range(3):
    plt.plot(wavelengths,residual_svd[0][:, i])

spectra = specfit_result.e_matrix('dataset1')
plt.subplot(3, 3, 2)
plt.title('EAS')
for i in range(spectra.shape[1]):
    plt.plot(wavelengths, spectra[:,i])

plt.tight_layout()
plt.show()

concentrations = specfit_result.c_matrix('dataset1')
