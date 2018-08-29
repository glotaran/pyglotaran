import numpy as np

def create_group(model, group_axis='estimated', xtol=0.5):
    group = {}

    for _, dataset_descriptor in model.dataset.items():
        if dataset_descriptor.dataset is None:
            raise Exception("Missing data for dataset '{dataset_descriptor.label}'")
        axis = dataset_descriptor.dataset.get_estimated_axis() if group_axis == 'estimated' \
            else dataset_descriptor.dataset.get_calculated_axis()
        axis = dataset_descriptor.dataset.get_axis(axis)
        for index in axis:
            group_index = index if not any(abs(index-val) < xtol for val in group) \
                else [val for val in group if abs(index-val) < xtol][0]
            if group_index not in group:
                group[group_index] = []
            group[group_index].append((index, dataset_descriptor))
    return group

def calculate_group(group, model, parameter, matrix='calculated'):

    matrix_func = model.estimated_matrix if matrix == 'estimated' else model.calculated_matrix
    if matrix_func is None:
        raise Exception("Missing function for '{matrix}' matrix")

    result = []
    for _, item in group.items():
        full = None
        full_compartments = None
        for index, dataset_descriptor in item:

            if dataset_descriptor.dataset is None:
                raise Exception("Missing data for dataset '{dataset_descriptor.label}'")

            axis = dataset_descriptor.dataset.get_estimated_axis() if matrix == 'estimated' \
                else dataset_descriptor.dataset.get_calculated_axis()
            axis = dataset_descriptor.dataset.get_axis(axis)

            dataset_descriptor = dataset_descriptor.fill(model, parameter)

            (compartments, this_matrix) = matrix_func(dataset_descriptor, index, axis)

            if full is None:
                full = this_matrix
                full_compartments = compartments
            else:
                if not compartments == full_compartments:
                    for comp in compartments:
                        if comp not in full_compartments:
                            full_compartments.append(comp)
                            full = np.concatenate((full, np.zeros((full.shape[1]))))
                    reshape = np.zeros((len(full_compartments), this_matrix.shape[1]))
                    for i, comp in enumerate(full_compartments):
                        reshape[i, :] = this_matrix[compartments.index(comp), :] \
                                if comp in compartments else np.zeros((this_matrix.shape[1]))
                    this_matrix = reshape

                full = np.concatenate((full, this_matrix), axis=1)
        result.append(full)
    return result
