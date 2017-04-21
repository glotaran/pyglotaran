from glotaran.io.mlsd_file_format import MLSDFile
from glotaran.plotting.glotaran_color_codes import get_glotaran_default_colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg

data_file_mlsd = MLSDFile('../resources/Al_6_1.txt')
data_file_mlsd.read("AL_6_1")
dataset_te = data_file_mlsd.dataset()

xmin = min(dataset_te.get_axis("time"))
xmax = max(dataset_te.get_axis("time"))
ymin = 650 # min(dataset_te.get_axis("spec"))
ymax = 800 # max(dataset_te.get_axis("spec"))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
plt.tight_layout()

ax1.pcolormesh(dataset_te.get_axis("time"),dataset_te.get_axis("spec"), dataset_te.data)
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymin, ymax])
ax1.set_ylabel('Wavelength (nm)')

U1, s1, V1 = linalg.svd(dataset_te.data)
print("U.shape: {} ; V.shape: {} ; s.shape: {}".format(U1.shape, V1.shape, s1.shape))
ax2.plot(dataset_te.get_axis("time"),V1[0:3,:].T)
ax2.set_xlim([xmin, xmax])
ax3.plot(dataset_te.get_axis("spec"),U1[:,0:3])
ax3.set_xlim([ymin, ymax])

data_file_mlsd_cmp = pd.read_csv('../resources/Al_6_1.csv', header=None,index_col=None)
times_cmp = data_file_mlsd_cmp.values[1:,0]
wavelength_cmp = data_file_mlsd_cmp.values[0,1:]
data_cmp = data_file_mlsd_cmp.values[1:,1:].T

ax4.set_xlim([xmin, xmax])
ax4.set_ylim([ymin, ymax])
ax4.pcolormesh(times_cmp, wavelength_cmp, data_cmp)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Wavelength (nm)')
# print(len(dataset_te.get_axis("time")))
# print(range(1,len(dataset_te.get_axis("time"))))
# plt.xticks(list(range(1,len(dataset_te.get_axis("time")))), dataset_te.get_axis("time"))
# plt.xscale('symlog', linthreshx=20)


# plt.ylabel('Time (ps)')
# plt.xlabel('$Wavenumber\ [\ cm^{-1}\ ]$')
# plt.pcolormesh(dataset_te.get_axis("spec"),dataset_te.get_axis("time"), dataset_te.data.T)

data_file_mlsd.write('test.ascii',overwrite=True)
plt.show()

