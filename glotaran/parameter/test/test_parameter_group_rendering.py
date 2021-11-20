from IPython.core.formatters import format_display_data

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

RENDERED_MARKDOWN = """\
  * __irf__:

    | _Label_   |   _Value_ |   _StdErr_ |   _Min_ |   _Max_ | _Vary_   | _Non-Negative_   | _Expr_   |
    |-----------|-----------|------------|---------|---------|----------|------------------|----------|
    | center    |       1.3 |          0 |    -inf |     inf | True     | False            | None     |
    | width     |       7.8 |          0 |    -inf |     inf | True     | False            | None     |

  * __j__:

    |   _Label_ |   _Value_ |   _StdErr_ |   _Min_ |   _Max_ | _Vary_   | _Non-Negative_   | _Expr_   |
    |-----------|-----------|------------|---------|---------|----------|------------------|----------|
    |         1 |         1 |          0 |    -inf |     inf | False    | False            | None     |

  * __kinetic__:

    |   _Label_ |   _Value_ |   _StdErr_ |   _Min_ |   _Max_ | _Vary_   | _Non-Negative_   | _Expr_   |
    |-----------|-----------|------------|---------|---------|----------|------------------|----------|
    |         1 |     0.3   |          0 |    -inf |     inf | True     | False            | None     |
    |         2 |     0.05  |          0 |    -inf |     inf | True     | False            | None     |
    |         3 |     0.007 |          0 |    -inf |     inf | True     | False            | None     |

"""  # noqa: E501


def test_param_group_markdown_is_order_independent():
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

    initial_parameters1 = load_parameters(PARAMETERS_3C_INITIAL1, format_name="yml_str")
    initial_parameters2 = load_parameters(PARAMETERS_3C_INITIAL2, format_name="yml_str")

    assert initial_parameters1.markdown() == RENDERED_MARKDOWN
    assert initial_parameters2.markdown() == RENDERED_MARKDOWN
    assert initial_parameters_ref.markdown() == RENDERED_MARKDOWN


def test_param_group_repr():
    """Repr creates code to recreate the object with from_dict."""
    result = ParameterGroup.from_dict({"foo": {"bar": [["1", 1.0], ["2", 2.0], ["3", 3.0]]}})
    result_short = ParameterGroup.from_dict({"foo": {"bar": [1, 2, 3]}})
    expected = "ParameterGroup.from_dict({'foo': {'bar': [['1', 1.0], ['2', 2.0], ['3', 3.0]]}})"

    assert result == result_short
    assert result_short.__repr__() == expected
    assert result.__repr__() == expected
    assert result == eval(result.__repr__())


def test_param_group_repr_from_list():
    """Repr creates code to recreate the object with from_list."""
    result = ParameterGroup.from_list([["1", 2.3], ["2", 3.0]])
    result_short = ParameterGroup.from_list([2.3, 3.0])
    expected = "ParameterGroup.from_list([['1', 2.3], ['2', 3.0]])"

    assert result == result_short
    assert result.__repr__() == expected
    assert result_short.__repr__() == expected
    assert result == eval(result.__repr__())


def test_param_group_ipython_rendering():
    """Autorendering in ipython"""
    param_group = ParameterGroup.from_dict({"foo": {"bar": [["1", 1.0], ["2", 2.0], ["3", 3.0]]}})
    rendered_obj = format_display_data(param_group)[0]

    assert "text/markdown" in rendered_obj
    assert rendered_obj["text/markdown"].startswith("  * __foo__")

    rendered_markdown_return = format_display_data(param_group.markdown())[0]

    assert "text/markdown" in rendered_markdown_return
    assert rendered_markdown_return["text/markdown"].startswith("  * __foo__")
