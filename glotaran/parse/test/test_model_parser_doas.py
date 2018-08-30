from unittest import TestCase
from glotaran.specification_parser import parse_file
from glotaran.models.damped_oscillation import (DOASModel,
                                                DOASMegacomplex,
                                                Oscillation)

# unused import
# from glotaran.model import FixedConstraint, BoundConstraint
# from os import getcwd
from os.path import join, dirname, abspath

THIS_DIR = dirname(abspath(__file__))


class TestParser(TestCase):

    def setUp(self):
        spec_path = join(THIS_DIR, 'test_model_spec_doas.yml')
        self.model = parse_file(spec_path)

    def test_print_model_spec(self):
        print(self.model)

    def test_compartments(self):
        self.assertTrue(isinstance(self.model.compartments, list))
        self.assertEqual(self.model.compartments, ['os1'])

    def test_model_type(self):
        self.assertTrue(isinstance(self.model, DOASModel))

    def test_oscillation(self):
        self.assertEqual(len(self.model.oscillations), 2)

        i = 1
        for _ in self.model.oscillations:
            label = "osc{}".format(i)
            self.assertTrue(label in self.model.oscillations)
            oscillation = self.model.oscillations[label]
            self.assertTrue(isinstance(oscillation, Oscillation))
            self.assertTrue(oscillation.label, label)
            self.assertEqual(oscillation.compartment, f"os{i}")
            self.assertEqual(oscillation.frequency, i)
            self.assertEqual(oscillation.rate, 2+i)

            i = i + 1

    def test_megacomplexes(self):
        self.assertEqual(len(self.model.megacomplexes), 4)

        i = 1
        for _ in self.model.megacomplexes:
            label = "cmplx{}".format(i)
            self.assertTrue(label in self.model.megacomplexes)
            megacomplex = self.model.megacomplexes[label]
            self.assertTrue(isinstance(megacomplex, DOASMegacomplex))
            self.assertEqual(megacomplex.label, label)
            self.assertEqual(megacomplex.k_matrices, ["km{}".format(i)])
            if i is 2:
                self.assertEqual(megacomplex.oscillations, ["osc1"])
            if i is 4:
                self.assertEqual(megacomplex.oscillations, ["osc2"])

            i = i + 1
