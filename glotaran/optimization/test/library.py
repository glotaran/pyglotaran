from glotaran.optimization.test.models import TestModelExponential
from glotaran.optimization.test.models import TestModelGaussian

test_library = {
    "decay_independent": TestModelExponential(
        type="test-model-exponential",
        label="decay_independent",
        is_index_dependent=False,
        compartments=["c1", "c2"],
        rates=["rates.decay.1", "rates.decay.2"],
    ),
    "decay_dependent": TestModelExponential(
        type="test-model-exponential",
        label="decay_dependent",
        is_index_dependent=True,
        compartments=["c1", "c2"],
        rates=["rates.decay.1", "rates.decay.2"],
    ),
    "gaussian": TestModelGaussian(
        type="test-model-gaussian",
        label="gaussian",
        compartments=["c1", "c2"],
        amplitude=["gaussian.amplitude.1", "gaussian.amplitude.2"],
        location=["gaussian.location.1", "gaussian.location.2"],
        width=["gaussian.width.1", "gaussian.width.2"],
    ),
}
