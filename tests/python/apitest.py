from glotaran import load, parse
from unittest import TestCase
from os import getcwd
from os.path import join, dirname, abspath

THIS_DIR = dirname(abspath(__file__))


class TestParser(TestCase):

    def setUp(self):
        print(getcwd())
        self.ref_spec_path = join(THIS_DIR, '../resources/models/test_model_spec.yml')
        self.spec_path = join(THIS_DIR, '../resources/models/test_glotaran_api_spec.yml')

        self.fitspec = '''
        type: kinetic

        parameters:
          - 0.1
          - 0.2

        compartments: [s1, s2]

        megacomplexes:
            - label: mc1
              k_matrices: [k1]

        k_matrices:
          - label: "k1"
            type: sequential
            compartments: [s1, s2, s3]
            parameters: [1, 2, 3]
            matrix: {
              '("s1","s1")': 1,
              '("s2","s2")': 2,
            }

        irf:
          - label: irf1
            type: gaussian
            center: 3
            width: 4

        datasets:
          - label: dataset1
            megacomplexes: [mc1]
            irf: irf1
        '''

    def test_reference_model_spec_from_file(self):
        print("Loading from file")
        gta0 = load(self.ref_spec_path)
        print(gta0)

    def test_fitspec_from_string(self):
        gta1 = parse(self.fitspec)
        print("Parsing from string")
        print(gta1.k_matrices)
        print(gta1)
        #TODO: Replace with another test:
        # gta1.eval()

    def test_fitspec_from_file(self):
        print("Loading from file")
        gta2 = load(self.spec_path)
        print(gta2)



