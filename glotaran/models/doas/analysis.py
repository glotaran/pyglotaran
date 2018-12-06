import numpy as np


def retrieve_doas(result, dataset):
    labels, clp = result.get_clp(dataset)
    dataset = result.model.dataset[dataset].fill(result.model, result.best_fit_parameter)
    result = {}
    for cmplx in dataset.megacomplex:
        oscillations = cmplx.oscillation
        oscillations = [osc.label for osc in oscillations]
        dim1 = clp.shape[0]
        dim2 = len(oscillations)
        doas = np.zeros((dim1, dim2), dtype=np.float64)
        for i, osc in enumerate(oscillations):
            sin = clp[:, labels.index(f'{osc}_sin')]
            cos = clp[:, labels.index(f'{osc}_cos')]
            doas[:, i] = np.sqrt(sin*sin+cos*cos)
        result[cmplx. label] = (oscillations, doas)
    return result


def retrieve_phase(result, dataset):
    labels, clp = result.get_clp(dataset)
    dataset = result.model.dataset[dataset].fill(result.model, result.best_fit_parameter)
    result = {}
    for cmplx in dataset.megacomplex:
        oscillations = cmplx.oscillation
        oscillations = [osc.label for osc in oscillations]
        dim1 = clp.shape[0]
        dim2 = len(oscillations)
        phase = np.zeros((dim1, dim2), dtype=np.float64)
        for i, osc in enumerate(oscillations):
            sin = clp[:, labels.index(f'{osc}_sin')]
            cos = clp[:, labels.index(f'{osc}_cos')]
            phase[:, i] = np.unwrap(np.arctan2(cos, sin))

        result[cmplx. label] = (oscillations, phase)
    return result
