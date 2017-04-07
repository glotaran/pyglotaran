from glotaran.datasets.mlsd_file_format import MLSDFile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_file_mlsd = MLSDFile('../resources/Al_6_1.txt')
data_file_mlsd.read("AL_6_1")
dataset_te = data_file_mlsd.dataset()

xmin = min(dataset_te.get_axis("time"))
xmax = max(dataset_te.get_axis("time"))
ymin = 650 # min(dataset_te.get_axis("spec"))
ymax = 800 # max(dataset_te.get_axis("spec"))

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.pcolormesh(dataset_te.get_axis("time"),dataset_te.get_axis("spec"), dataset_te.data)
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymin, ymax])

data_file_mlsd_cmp = pd.read_csv('../resources/Al_6_1.csv', header=None,index_col=None)
times_cmp = data_file_mlsd_cmp.values[1:,0]
wavelength_cmp = data_file_mlsd_cmp.values[0,1:]
data_cmp = data_file_mlsd_cmp.values[1:,1:].T

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin, ymax])
ax2.pcolormesh(times_cmp, wavelength_cmp, data_cmp)

# print(len(dataset_te.get_axis("time")))
# print(range(1,len(dataset_te.get_axis("time"))))
# plt.xticks(list(range(1,len(dataset_te.get_axis("time")))), dataset_te.get_axis("time"))
# plt.xscale('symlog', linthreshx=20)


# plt.ylabel('Time (ps)')
# plt.xlabel('$Wavenumber\ [\ cm^{-1}\ ]$')
# plt.pcolormesh(dataset_te.get_axis("spec"),dataset_te.get_axis("time"), dataset_te.data.T)

data_file_mlsd.write('test.ascii',overwrite=True)
plt.show()

