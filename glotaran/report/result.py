
import holoviews as hv

from glotaran.analysis.fitresult import FitResult


def parameter(result: FitResult) -> hv.Curve:
    labels = []
    initials = []
    optimized = []
    for label, param in result.best_fit_parameter.all_with_label():
        labels.append(label)
        initials.append(f'{result.initial_parameter.get(label).value:0.8e}')
        optimized.append(f'{param.value:0.8e}')
    return hv.Table((labels, initials, optimized), kdims=['Label', 'Initital', 'Optimized'])\
        .options(max_rows=len(labels), max_value_len=100)


def result(result: FitResult) -> hv.Curve:
    labels = [
        "Non-linear least squares",
        "Number Function Evaluations",
        "Number of variables",
        "Number of data points",
        "Degrees of freedom",
        "Chi Square",
        "Chi Square Reduced",
    ]
    values = [
        result.nnls,
        result.nfev,
        result.nvars,
        result.ndata,
        result.nfree,
        f'{result.initial_parameter.get(label).value:0.8e}'
    ]
    optimized = []
    for label, param in result.best_fit_parameter.all_with_label():
        labels.append(label)
        initials.append(result.initial_parameter.get(label).value)
        optimized.append(param.value)
    return hv.Table((labels, initials, optimized), kdims=['Label', 'Initital', 'Optimized'])

