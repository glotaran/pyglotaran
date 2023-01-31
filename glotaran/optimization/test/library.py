from glotaran.model.library import Library
from glotaran.optimization.test.models import TestMegacomplexExponential
from glotaran.optimization.test.models import TestMegacomplexGaussian

TestLibrary = Library.create_for_megacomplexes(
    [TestMegacomplexExponential, TestMegacomplexGaussian]
)(
    megacomplex={
        "decay_independent": {
            "type": "test-megacomplex-exponential",
            "is_index_dependent": False,
            "compartments": ["c1", "c2"],
            "rates": ["rates.decay.1", "rates.decay.2"],
        },
        "decay_dependent": {
            "type": "test-megacomplex-exponential",
            "is_index_dependent": True,
            "compartments": ["c1", "c2"],
            "rates": ["rates.decay.1", "rates.decay.2"],
        },
        "gaussian": {
            "type": "test-megacomplex-gaussian",
            "compartments": ["c1", "c2"],
            "amplitude": ["gaussian.amplitude.1", "gaussian.amplitude.2"],
            "location": ["gaussian.location.1", "gaussian.location.2"],
            "width": ["gaussian.width.1", "gaussian.width.2"],
        },
    }
)
