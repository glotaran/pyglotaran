from glotaran.datasets.wavelength_time_explicit_file import WavelengthExplicitFile, TimeExplicitFile, ExplicitFile
import numpy as np
import matplotlib.pyplot as plt

test = ' Time explicit\n'
print(not test.lower().find("time"))

data_file_te = ExplicitFile('../resources/te_data_file.ascii')

data_file_te.read("psi_1")
dataset_te = data_file_te.dataset()

print(dataset_te.get_axis("time"))
print(dataset_te.get_axis("spec"))
# print(dataset_te.data)
# axis(v)
# v = [xmin, xmax, ymin, ymax]

fig = plt.figure()
plt.xlabel('Time (ps)')
plt.axis([min(dataset_te.get_axis("time")),max(dataset_te.get_axis("time")),min(dataset_te.get_axis("spec")),max(dataset_te.get_axis("spec"))])
plt.ylabel('$Wavenumber\ [\ cm^{-1}\ ]$')
plt.pcolormesh(dataset_te.get_axis("time"),dataset_te.get_axis("spec"), dataset_te.data)
print(len(dataset_te.get_axis("time")))
print(range(1,len(dataset_te.get_axis("time"))))
plt.xticks(list(range(1,len(dataset_te.get_axis("time")))), dataset_te.get_axis("time"))
plt.xscale('symlog', linthreshx=20)


# plt.ylabel('Time (ps)')
# plt.xlabel('$Wavenumber\ [\ cm^{-1}\ ]$')
# plt.pcolormesh(dataset_te.get_axis("spec"),dataset_te.get_axis("time"), dataset_te.data.T)
plt.show()
data_file_te.write('test.ascii')