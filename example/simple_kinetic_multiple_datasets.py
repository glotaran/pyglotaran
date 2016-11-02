from glotaran_tools import SimulatedSpectralTimetrace
from glotaran_core import (KineticModel, KineticParameter, GaussianIrf,
                           Models, Datasets, GlobalAnalysis)

set1 = SimulatedSpectralTimetrace([1, 2.5], [1e-3, 2e-5], [100, 200], [10, 10],
                                  50, 250, 1, 500, 1, label="set1")
set2 = SimulatedSpectralTimetrace([1, 2.5], [1e-3, 5e-5], [100, 200], [10, 10],
                                  50, 250, 1, 500, 1, label="set2")

model1 = KineticModel("set1", [KineticParameter("k1", 1e-3),
                               KineticParameter("k2", 2e-3)],
                      Irf(1, 150))
model2 = model1.derive("set2", [KineticParameter("k2", 5e-5)])

analysis = GlobalAnalysis(Datasets([set1, set2]), Models([model1, model2]))
