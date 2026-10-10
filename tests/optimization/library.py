from __future__ import annotations

from tests.optimization.elements import TestElementConstant
from tests.optimization.elements import TestElementExponential
from tests.optimization.elements import TestElementGaussian

test_library = {
    "decay_independent": TestElementExponential(
        type="test-element-exponential",
        label="decay_independent",
        is_index_dependent=False,
        compartments=["c1", "c2"],
        rates=["rates.decay.1", "rates.decay.2"],
    ),
    "decay_dependent": TestElementExponential(
        type="test-element-exponential",
        label="decay_dependent",
        is_index_dependent=True,
        compartments=["c1", "c2"],
        rates=["rates.decay.1", "rates.decay.2"],
    ),
    "gaussian": TestElementGaussian(
        type="test-element-gaussian",
        label="gaussian",
        compartments=["c1", "c2"],
        amplitude=["gaussian.amplitude.1", "gaussian.amplitude.2"],
        location=["gaussian.location.1", "gaussian.location.2"],
        width=["gaussian.width.1", "gaussian.width.2"],
    ),
    "constant": TestElementConstant(
        type="test-element-constant",
        label="constant",
        dimension="model",
        compartments=["cc"],
        value=2,
        is_index_dependent=False,
    ),
}
