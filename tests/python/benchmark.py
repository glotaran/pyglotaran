import timeit
import time
import sys

import numpy as np

bench_start = time.time()


from glotaran.specification_parser import parse_yml


fitspec = '''
type: kinetic

parameters: {}

compartments: [s1, s2, s3]

megacomplexes:
    - label: mc1
      k_matrices: [k1]

k_matrices:
  - label: "k1"
    matrix: {{
      '("s1","s1")': 1,
      '("s2","s2")': 2,
      '("s3","s3")': 3,
}}

shapes:
  - label: "shape1"
    type: "gaussian"
    amplitude: shape.amps.1
    location: shape.locs.1
    width: shape.width.1
  - label: "shape2"
    type: "gaussian"
    amplitude: shape.amps.2
    location: shape.locs.2
    width: shape.width.2
  - ["shape3", "gaussian", shape.amps.3, shape.locs.3, shape.width.3]

initial_concentrations: []

irf: []

datasets:
  - label: dataset1
    type: spectral
    megacomplexes: [mc1]
    path: ''
    shapes:
      - compartment: s1
        shape: shape1
      - [s2, shape2]
      - [s3, shape3]

'''

params = [101e-3, 202e-4, 305e-5]
times = np.asarray(np.arange(0, 1500, 1.5))
x = np.arange(12820, 15120, 4.6)
amps = [7, 3, 30, False]
locations = [14700, 13515, 14180, False]
delta = [400, 100, 300, False]

params.append({'shape': [{'amps': amps}, {'locs': locations},
                         {'width': delta}]})

model = parse_yml(fitspec.format(params))

axies = {"time": times, "spectral": x}


def bench_eval(n):
    print("Benchmarking Dataset Simulation")

    def f():
        model.eval('dataset1', axies)
    t = timeit.timeit(f, number=n) / n
    print("Result: {}".format(t))
    return t


def bench_c_matrix(n):
    print("Benchmarking c_matrix")

    def f():
        model.fit_model().c_matrix(model.parameter.as_parameters_dict())
    t = timeit.timeit(f, number=n) / n
    print("Result: {}".format(t))
    return t


def bench_e_matrix(n):
    print("Benchmarking e_matrix")

    def f():
        model.fit_model().e_matrix(model.parameter.as_parameters_dict(),
                                   **{'dataset': 'dataset1'})
    t = timeit.timeit(f, number=n) / n
    print("Result: {}".format(t))
    return t


def bench_residual(n):
    print("Benchmarking Residual")

    def f():
        model.fit_model().result()._residual(model.parameter.
                                             as_parameters_dict())
    t = timeit.timeit(f, number=n) / n
    print("Result: {}".format(t))
    return t


REPORT = '''Benchmark Results

Iterations: {}

Model Evaluation:\t{:.6f}
C Matrix Calculation:\t{:.6f}
E Matrix Calculation:\t{:.6f}
Residual Calculation:\t{:.6f}

Total:\t\t\t {:.6f}
'''


if __name__ == "__main__":
    out = None
    n = 2
    for arg in sys.argv[1:]:
        if "-n=" in arg:
            n = int(arg[3:])
        if "-o=" in arg:
            out = arg[3:]

    print("Starting Glotaran Benchmark")
    print("Nr Loops: {}".format(n))
    t_s = bench_eval(n)
    t_c = bench_c_matrix(n)
    t_e = bench_e_matrix(n)
    t_r = bench_residual(n)
    t = time.time() - bench_start
    print("")
    rep = REPORT.format(n, t_s, t_c, t_e, t_r, t)
    print(rep)

    if out is not None:
        print("Writing result to '{}'".format(out))
        with open(out, 'w') as f:
            f.write(rep)
