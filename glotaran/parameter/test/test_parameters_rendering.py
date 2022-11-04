from IPython.core.formatters import format_display_data

from glotaran.io import load_parameters
from glotaran.parameter.parameters import Parameters

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
    - ["2", 500e-4, {standard-error: 0.000012345678}]
    - ["3", {expr: $kinetic.1 + $kinetic.2}]
"""

RENDERED_MARKDOWN = """\
  * __irf__:

    | _Label_   |   _Value_ |   _Standard Error_ | _t-value_   |   _Minimum_ |   _Maximum_ | _Vary_   | _Non-Negative_   | _Expression_   |
    |-----------|-----------|--------------------|-------------|-------------|-------------|----------|------------------|----------------|
    | center    | 1.300e+00 |                nan |  nan        |        -inf |         inf | True     | False            | `None`         |
    | width     | 7.800e+00 |                nan |  nan        |        -inf |         inf | True     | False            | `None`         |

  * __j__:

    |   _Label_ |   _Value_ |   _Standard Error_ | _t-value_   |   _Minimum_ |   _Maximum_ | _Vary_   | _Non-Negative_   | _Expression_   |
    |-----------|-----------|--------------------|-------------|-------------|-------------|----------|------------------|----------------|
    |         1 | 1.000e+00 |                nan |  nan        |        -inf |         inf | False    | False            | `None`         |

  * __kinetic__:

    |   _Label_ |   _Value_ |   _Standard Error_ | _t-value_   |   _Minimum_ |   _Maximum_ | _Vary_   | _Non-Negative_   | _Expression_              |
    |-----------|-----------|--------------------|-------------|-------------|-------------|----------|------------------|---------------------------|
    |         1 | 3.000e-01 |        nan         |  nan        |        -inf |         inf | True     | False            | `None`                    |
    |         2 | 5.000e-02 |          1.235e-05 |  4050       |        -inf |         inf | True     | False            | `None`                    |
    |         3 | 3.500e-01 |        nan         |  nan        |        -inf |         inf | False    | False            | `$kinetic.1 + $kinetic.2` |

"""  # noqa: E501

RENDERED_MARKDOWN_E5_PRECISION = """\
  * __irf__:

    | _Label_   |     _Value_ |   _Standard Error_ | _t-value_   |   _Minimum_ |   _Maximum_ | _Vary_   | _Non-Negative_   | _Expression_   |
    |-----------|-------------|--------------------|-------------|-------------|-------------|----------|------------------|----------------|
    | center    | 1.30000e+00 |        1.23457e-05 |  105300     |        -inf |         inf | True     | False            | `None`         |

"""  # noqa: E501


def test_parameters_markdown_is_order_independent():
    """Markdown output of Parameters.markdown() is independent of initial order"""
    PARAMETERS_3C_INITIAL1 = f"""{PARAMETERS_3C_BASE}\n{PARAMETERS_3C_KINETIC}"""
    PARAMETERS_3C_INITIAL2 = f"""{PARAMETERS_3C_KINETIC}\n{PARAMETERS_3C_BASE}"""

    initial_parameters_ref = Parameters.from_dict(
        {
            "j": [["1", 1, {"vary": False, "non-negative": False}]],
            "kinetic": [
                ["1", 0.3],
                ["2", 500e-4, {"standard-error": 0.000012345678}],
                ["3", 700e-5, {"expr": "$kinetic.1 + $kinetic.2"}],
            ],
            "irf": [["center", 1.3], ["width", 7.8]],
        }
    )

    initial_parameters1 = load_parameters(PARAMETERS_3C_INITIAL1, format_name="yml_str")
    initial_parameters2 = load_parameters(PARAMETERS_3C_INITIAL2, format_name="yml_str")

    assert str(initial_parameters1.markdown()) == RENDERED_MARKDOWN
    assert str(initial_parameters2.markdown()) == RENDERED_MARKDOWN
    assert str(initial_parameters_ref.markdown()) == RENDERED_MARKDOWN

    minimal_params = Parameters.from_dict(
        {"irf": [["center", 1.3, {"standard-error": 0.000012345678}]]}
    )

    assert str(minimal_params.markdown(float_format=".5e")) == RENDERED_MARKDOWN_E5_PRECISION


def test_parameters_repr():
    """Repr creates code to recreate the object with from_dict."""

    # Needed to eval the Parameters repr
    from glotaran.parameter.parameter import Parameter  # noqa:401

    result = Parameters.from_dict(
        {
            "foo": {
                "bar": [
                    ["1", 1.0, {"vary": True}],
                    ["2", 2.0, {"expression": "$foo.bar.1*2"}],
                    ["3", 3.0, {"min": -10}],
                ]
            }
        }
    )
    expected = (
        "Parameters({'foo.bar.1': Parameter(label='foo.bar.1', value=1.0), "
        "'foo.bar.2': Parameter(label='foo.bar.2', value=2.0, expression='$foo.bar.1*2',"
        " vary=False), "
        "'foo.bar.3': Parameter(label='foo.bar.3', value=3.0, minimum=-10)})"
    )

    print(result.__repr__())
    assert result.__repr__() == expected
    assert result == eval(expected)


def test_parameters_ipython_rendering():
    """Autorendering in ipython"""
    param_group = Parameters.from_dict({"foo": {"bar": [["1", 1.0], ["2", 2.0], ["3", 3.0]]}})
    rendered_obj = format_display_data(param_group)[0]

    assert "text/markdown" in rendered_obj
    assert rendered_obj["text/markdown"].startswith("  * __foo__")

    rendered_markdown_return = format_display_data(param_group.markdown())[0]

    assert "text/markdown" in rendered_markdown_return
    assert rendered_markdown_return["text/markdown"].startswith("  * __foo__")
