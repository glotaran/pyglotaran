import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

from glotaran.io.wavelength_time_explicit_file import ExplicitFile
from glotaran.plotting.basic_plots import plot_data
from glotaran.specification_parser import parse_yml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_path = os.path.join(THIS_DIR, '..', 'resources', 'data', 'streakdata.ascii')

# Settings:
reproduce_figures_from_paper = True
# Read in streakdata.ascii from resources/data sub-folder
data_file_te = ExplicitFile(root_data_path)
dataset_te = data_file_te.read("dataset1")
#dataset_te = data_file_te.dataset()
times = dataset_te.get_axis("time")
times = list(np.asarray(times) + 83)
wavelengths = dataset_te.get_axis("spectral")

# Get data limits
if reproduce_figures_from_paper:
    [xmin, xmax] = [-20, 200] #with respect to maximum of IRF (needs function written)
    [ymin, ymax] = [630,770]
    linear_range = [-20, 20]
else:
    [xmin,xmax] = [min(dataset_te.get_axis("time")), max(dataset_te.get_axis("time"))]
    [ymin, ymax] = [min(dataset_te.get_axis("spec")),max(dataset_te.get_axis("spec"))]
    linear_range = [-20, 20]
print([xmin,xmax,ymin,ymax])

# Plot the data
fig, all_axes = plt.subplots(2, 2)

axData = all_axes[0,0]
plot_data(axData, times, wavelengths, dataset_te.data)
plt.show(block=False)

axDataMesh = all_axes[0,1]
meshLin = axDataMesh.pcolormesh(times, wavelengths, dataset_te.data, cmap='Greys')
axDataMesh.set_xscale('linear')
axDataMesh.spines['right'].set_visible(True)
axDataMesh.yaxis.set_ticks_position('left')
axDataMesh.set_xlim((linear_range[0], linear_range[1]))
axDataMesh.set_ylim(ymin, ymax)
axDataMesh.yaxis.set_ticks_position('left')
axDataMesh.yaxis.set_visible(True)

divider = make_axes_locatable(axDataMesh)
axLog = divider.append_axes("right", size="50%", pad=0, sharey=axDataMesh)
plt.setp(axDataMesh.get_xticklabels(), visible=True)
meshLog = axLog.pcolormesh(times, wavelengths, dataset_te.data, cmap='Greys')
axLog.set_xscale('log')
axLog.set_xlim((linear_range[1], xmax))
axLog.xaxis.set_ticks_position('bottom')
axLog.spines['left'].set_visible(False)
axLog.yaxis.set_visible(False)
axLog.set_ylim(ymin, ymax)
axLog.yaxis.set_ticks_position('right')
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig1 = axLog.get_figure()
fig1.add_axes(ax_cb)
plt.colorbar(meshLog, cax=ax_cb)
#fig.colorbar(meshLin, pad=20.2)
plt.show(block=False)
#ax2 = axLog.twinx()
#ax2.spines['right'].set_visible(False)
#ax2.tick_params(axis='y',which='both',labelright='on')

axDataContourf = all_axes[1,0]
levels = np.linspace(0, max(dataset_te.data.flatten()), 10)
cnt = axDataContourf.contourf(times, wavelengths, dataset_te.data, levels=levels, cmap="Greys")
# This is the fix for the white lines between contour levels
for c in cnt.collections:
    c.set_edgecolor("face")

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
 - 13200.0

compartments: [s1, s2, s3, s4]

megacomplexes:
    - label: mc1
      k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {
      '("s1","s1")': 3,
      '("s2","s2")': 4,
      '("s3","s3")': 5,
      '("s4","s4")': 6
    }

irf:
  - label: irf
    type: gaussian
    center: 1
    width: 2
    backsweep: True
    backsweep_period: 7

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''
    irf: irf

'''

specfit_model = parse_yml(fitspec)
#print(specfit_model)
times = np.asarray(dataset_te.get_axis("time"))
wavelengths = np.asarray(dataset_te.get_axis("spectral"))
specfit_model.datasets['dataset1'].data = dataset_te
specfit_result = specfit_model.fit()
specfit_result.best_fit_parameter.pretty_print()

residual = specfit_result.final_residual()
axResidualContourf = all_axes[1,1]
levels = np.linspace(0, max(dataset_te.data.flatten()), 10)
cnt = axResidualContourf.contourf(times, wavelengths, residual, levels=levels, cmap="Greys")
# This is the fix for the white lines between contour levels
for c in cnt.collections:
    c.set_edgecolor("face")

plt.show()

# plot svd of residual
residual_svd = specfit_result.final_residual_svd()

# TODO: getting e_matrix still crashes
spectra = specfit_result.e_matrix('dataset1')
#TODO: spectra is a list of length(timepoints) - this is too much
plt.plot(wavelengths, spectra[0])