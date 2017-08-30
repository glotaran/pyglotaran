import matplotlib.pyplot as plt
import numpy as np
import os
from glotaran.plotting.glotaran_color_codes import get_glotaran_default_colors_cycler
from cycler import cycler

from glotaran.dataio.wavelength_time_explicit_file import ExplicitFile
from glotaran.specification_parser import parse_yml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_path = os.path.join(THIS_DIR, '..', 'resources', 'data')
datapath_PAL_open = os.path.join(root_data_path, 'PAL_700_ma_tr2_gain50_10uW_590nmfilter_21C_400nm_AV_bc_sh_sel_620_830.ascii')
datapath_PAL_closed = os.path.join(root_data_path, 'PAL_DCMU_80uM_WL_SF_700_ma_tr2_gain50_100uW_590nmfilter_21C_400nm_AV_bc_sh_sel_620_830.ascii')
prop_cycle=get_glotaran_default_colors_cycler()

# Read in data from resources/data sub-folder
# Dataset1
datafile_PAL_open = ExplicitFile(datapath_PAL_open)
dataset_PAL_open = datafile_PAL_open.read("dataset_PAL_open")
# Dataset2
datafile_PAL_closed = ExplicitFile(datapath_PAL_closed)
dataset_PAL_closed = datafile_PAL_closed.read("dataset_PAL_closed")

times1 = dataset_PAL_open.get_axis("time")
times2 = dataset_PAL_closed.get_axis("time")
wavelengths1 = dataset_PAL_open.get_axis("spectral")
wavelengths2 = dataset_PAL_closed.get_axis("spectral")

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
plt.figure(figsize=(12, 8))
plt.subplot(3, 4, 1)
plt.title('PAL_open')
plt.pcolormesh(times1, wavelengths1, dataset_PAL_open.data)
plt.subplot(3, 4, 5)
plt.title('PAL_closed')
plt.pcolormesh(times2, wavelengths2, dataset_PAL_closed.data)

rsvd1, svals1, lsvd1 = np.linalg.svd(dataset_PAL_open.data)
rsvd2, svals2, lsvd2 = np.linalg.svd(dataset_PAL_closed.data)
plt.subplot(3, 4, 2)
plt.title('LSV PAL open')
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())  # unsure why this is not working
for i in range(4):
    plt.plot(times1, lsvd1[i, :])
plt.subplot(3, 4, 6)
plt.title('LSV PAL closed')
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())  # because here it works
for i in range(4):
    plt.plot(times2, lsvd2[i, :])
# Plot singular values (SV)
plt.subplot(3, 4, 3)
plt.title('SVals PAL open')
plt.plot(range(max(10, min(len(times1), len(wavelengths1)))), svals1, 'ro')
plt.yscale('log')
plt.subplot(3, 4, 7)
plt.title('SVals PAL closed')
plt.plot(range(max(10, min(len(times2), len(wavelengths2)))), svals2, 'ro')
plt.yscale('log')
# Plot right singular vectors (RSV, wavelengths, first 3)
plt.subplot(3, 4, 4)
plt.title('RSV PAL open')
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())
for i in range(4):
    plt.plot(wavelengths1, rsvd1[:, i])
plt.subplot(3, 4, 8)
plt.title('RSV PAL closed')
plt.rc('axes', prop_cycle=get_glotaran_default_colors_cycler())
for i in range(4):
    plt.plot(wavelengths2, rsvd2[:, i])
plt.show(block=False)


plt.tight_layout()
plt.show()
