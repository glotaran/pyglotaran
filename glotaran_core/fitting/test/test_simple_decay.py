from unittest import TestCase
from glotaran_tools.specification_parser import parse_yml
from glotaran_core.model import Dataset
import numpy as np


test_parameter = [3.7e-3, 2]
test_amp = 5.6


class SimpleDecay(Dataset):
    def __init__(self, amp, rate):
        self._timepoints = np.asarray(np.arange(0, 1500, 1.5)).tolist()
        self._channel = []

        for i in range(len(self._timepoints)):
            self._channel.append(
                amp * np.exp(-self._timepoints[i] * rate)
            )

    def channels(self):
        return [self._channel]


class TestSimpleDecay(TestCase):

    def setUp(self):
        spec = '''type: kinetic

parameter: {}

initial_concentrations:
    - label: i1
      parameter: [2]

megacomplexes:
    - label: mc1
      k_matrices: [k1]

k_matrices:
  - label: "km1"
    matrix: {{
      '(1,1)': 1,
}}

irf: []

datasets: []

'''.format(test_parameter)
        self.model = parse_yml(spec)

    def test_simple_one_channel_decay(self):
        dataset = SimpleDecay(test_amp, test_parameter[1])
        print(self.model)
