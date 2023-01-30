from glotaran.model.library import Library
from glotaran.optimization.test.models import TestMegacomplexExponential
from glotaran.optimization.test.models import TestMegacomplexGaussian

TestLibrary = Library.create_for_megacomplexes(
    [TestMegacomplexExponential, TestMegacomplexGaussian]
)(
    megacomplex={
        "decay_independent": {
            "is_index_dependent": False,
            "compartments": ["c1", "c2", "c3"],
            "rates": ["rates.decay.1", "rates.decay.2", "rates.decay.3"],
        }
    }
)
