from glotaran.io import load_parameters
from glotaran.parameter.parameter_group import ParameterGroup

PARAMETERS_3C_BASE = """\
irf:
    - ["center", 1.3]
    - ["width", 7.8]
j:
    - ["1", 1, {"vary": False, "non-negative": False}]
"""

PARAMETERS_3C_KINETIC = """\
kinetic:
    - ["1", 300e-3]
    - ["2", 500e-4]
    - ["3", 700e-5]
"""


def test_markdown_is_order_idependent():
    """Markdown output of ParameterGroup.markdown() is independent of initial order"""
    PARAMETERS_3C_INITIAL1 = f"""{PARAMETERS_3C_BASE}\n{PARAMETERS_3C_KINETIC}"""
    PARAMETERS_3C_INITIAL2 = f"""{PARAMETERS_3C_KINETIC}\n{PARAMETERS_3C_BASE}"""

    initial_parameters_ref = ParameterGroup.from_dict(
        {
            "j": [["1", 1, {"vary": False, "non-negative": False}]],
            "kinetic": [
                ["1", 300e-3],
                ["2", 500e-4],
                ["3", 700e-5],
            ],
            "irf": [["center", 1.3], ["width", 7.8]],
        }
    )

    initial_parameters1 = load_parameters(PARAMETERS_3C_INITIAL1, fmt="yml_str")
    initial_parameters2 = load_parameters(PARAMETERS_3C_INITIAL2, fmt="yml_str")

    assert initial_parameters1.markdown() == initial_parameters_ref.markdown()
    assert initial_parameters2.markdown() == initial_parameters_ref.markdown()
