import collections
import itertools
import dask.array as da
import dask.bag as db
import numpy as np

ProblemDescriptor = collections.namedtuple('ProblemDescriptor',
                                           'dataset data matrix_axis global_axis')
GroupedProblem = collections.namedtuple('GroupedProblem', 'data descriptor')
GroupedProblemDescriptor = collections.namedtuple('ProblemDescriptor', 'dataset index axis')


def create_ungrouped_bag(scheme):
    bag = {}
    for label in scheme.model.dataset:
        bag[label] = ProblemDescriptor(
            scheme.model.dataset[label],
            scheme.data[label].data,
            scheme.data[label].coords[scheme.model.matrix_dimension],
            scheme.data[label].coords[scheme.model.global_dimension],
        )
    return bag


def create_grouped_bag(scheme):
    bag = None
    full_axis = None
    for label in scheme.model.dataset:
        dataset = scheme.data[label]
        global_axis = dataset.coords[scheme.model.global_dimension].values
        model_axis = dataset.coords[scheme.model.matrix_dimension].values
        if bag is None:
            bag = collections.deque(
                GroupedProblem(dataset.data.isel({scheme.model.global_dimension: i}),
                               [GroupedProblemDescriptor(label, value, model_axis)])
                for i, value in enumerate(global_axis)
            )
            full_axis = collections.deque(global_axis)
        else:
            i1, i2 = _find_overlap(full_axis, global_axis, atol=0.1)

            for i, j in enumerate(i1):
                bag[j] = GroupedProblem(
                    da.concatenate(
                        [bag[j][0], dataset.data.isel({scheme.model.global_dimension: i2[i]})]),
                    bag[j][1] + [GroupedProblemDescriptor(label, global_axis[i2[i]], model_axis)]
                )

            for i in range(0, i2[0]):
                full_axis.appendleft(global_axis[i2[i]])
                bag.appendleft(GroupedProblem(
                    dataset.data.isel({scheme.model.global_dimension: i2[i]}),
                    [GroupedProblemDescriptor(label, global_axis[i], model_axis)]
                ))

            for i in range(i2[-1]+1, len(global_axis)):
                full_axis.append(global_axis[i])
                bag.append(GroupedProblem(
                    dataset.data.isel({scheme.model.global_dimension: i2[i]}),
                    [GroupedProblemDescriptor(label, global_axis[i], model_axis)]
                ))

    return db.from_sequence(bag)


def _find_overlap(a, b, rtol=1e-05, atol=1e-08):
    ovr_a = []
    ovr_b = []
    start_b = 0
    for i, ai in enumerate(a):
        for j, bj in itertools.islice(enumerate(b), start_b, None):
            if np.isclose(ai, bj, rtol=rtol, atol=atol, equal_nan=False):
                ovr_a.append(i)
                ovr_b.append(j)
            elif bj > ai:  # (more than tolerance)
                break  # all the rest will be farther away
            else:  # bj < ai (more than tolerance)
                start_b += 1  # ignore further tests of this item
    return (ovr_a, ovr_b)
