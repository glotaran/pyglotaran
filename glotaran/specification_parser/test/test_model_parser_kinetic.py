from unittest import TestCase
from math import isnan, isinf
from glotaran.specification_parser import parse_file
from glotaran.models.spectral_temporal import (KineticModel,
                                               GaussianIrf,
                                               SpectralShapeGaussian,
                                               KineticMegacomplex)
from glotaran.model import (InitialConcentration,
                            ZeroConstraint,
                            EqualConstraint,
                            EqualAreaConstraint,
                            Parameter,
                            )

# unused import
# from glotaran.model import FixedConstraint, BoundConstraint
# from os import getcwd
from os.path import join, dirname, abspath
import numpy as np

THIS_DIR = dirname(abspath(__file__))


class TestParser(TestCase):

    def setUp(self):
        spec_path = join(THIS_DIR, 'test_model_spec_kinetic.yml')
        self.model = parse_file(spec_path)

    def test_print_model_spec(self):
        print(self.model)

    def test_compartments(self):
        self.assertTrue(isinstance(self.model.compartments, list))
        self.assertEqual(self.model.compartments, ['s1', 's2', 's3', 's4',
                                                   'osc1'])

    def test_model_type(self):
        self.assertTrue(isinstance(self.model, KineticModel))

    def test_dataset(self):
        self.assertTrue(len(self.model.datasets) is 2)

        i = 1
        for _ in self.model.datasets:
            label = "dataset{}".format(i)
            self.assertTrue(label in self.model.datasets)
            dataset = self.model.datasets[label]
            self.assertTrue(dataset.label == label)
            self.assertTrue(dataset.megacomplexes ==
                            ["cmplx{}".format(i)])
            self.assertTrue(dataset.initial_concentration ==
                            "inputD{}".format(i))
            self.assertTrue(dataset.irf == "irf{}".format(i))
            self.assertEqual(dataset.scaling, i)

            self.assertEqual(len(dataset.megacomplex_scaling), 2)

            self.assertTrue('cmplx1' in dataset.megacomplex_scaling)
            self.assertTrue('cmplx2' in dataset.megacomplex_scaling)
            for _, param in dataset.megacomplex_scaling.items():
                self.assertEqual(param, [1, 2])

            self.assertEqual(len(dataset.compartment_scaling), 2)

            self.assertTrue('s1' in dataset.compartment_scaling)
            self.assertTrue('s2' in dataset.compartment_scaling)
            for _, param in dataset.compartment_scaling.items():
                self.assertEqual(param, [3, 4])

            if i is 1:
                self.assertEqual(len(dataset.shapes), 2)
                self.assertTrue("s1" in dataset.shapes)
                self.assertEqual(dataset.shapes["s1"], ["shape1"])
                self.assertTrue("s2" in dataset.shapes)
                self.assertEqual(dataset.shapes["s2"], ["shape2"])

            else:
                self.assertTrue(len(dataset.compartment_constraints) is 4)

                self.assertTrue(any(isinstance(c, ZeroConstraint) for c in
                                    dataset.compartment_constraints))

                zcs = [zc for zc in dataset.compartment_constraints
                       if isinstance(zc, ZeroConstraint)]
                self.assertTrue(len(zcs) is 2)
                for zc in zcs:
                    self.assertEqual(zc.compartment, 's1')
                    self.assertEqual(zc.intervals, [(1, 100), (2, 200)])

                self.assertTrue(any(isinstance(c, EqualConstraint) for c in
                                    dataset.compartment_constraints))
                ec = [ec for ec in dataset.compartment_constraints
                      if isinstance(ec, EqualConstraint)][0]
                self.assertEqual(ec.compartment, 's2')
                self.assertEqual(ec.intervals, [(60, 700)])
                self.assertEqual(ec.targets, ['s1', 's2'])
                self.assertEqual(ec.parameters, [54, 56])

                self.assertTrue(any(isinstance(c, EqualAreaConstraint) for c in
                                    dataset.compartment_constraints))
                eac = [eac for eac in dataset.compartment_constraints
                       if isinstance(eac, EqualAreaConstraint)][0]
                self.assertEqual(eac.compartment, 's3')
                self.assertEqual(eac.intervals, [(670, 810)])
                self.assertEqual(eac.target, 's2')
                self.assertEqual(eac.parameter, 55)
                self.assertEqual(eac.weight, 0.0016)
            i = i + 1

    def test_initial_concentration(self):
        self.assertTrue(len(self.model.initial_concentrations) is 2)

        i = 1
        for _ in self.model.initial_concentrations:
            label = "inputD{}".format(i)
            self.assertTrue(label in self.model.initial_concentrations)
            initial_concentration = self.model.initial_concentrations[label]
            self.assertTrue(isinstance(initial_concentration,
                                       InitialConcentration))
            self.assertTrue(initial_concentration.label == label)
            self.assertTrue(initial_concentration.parameter == [1, 2, 3])

    def test_irf(self):
        self.assertTrue(len(self.model.irfs) is 2)

        i = 1
        for _ in self.model.irfs:
            label = "irf{}".format(i)
            self.assertTrue(label in self.model.irfs)
            irf = self.model.irfs[label]
            self.assertTrue(isinstance(irf, GaussianIrf))
            self.assertTrue(irf.label == label)
            want = [1] if i is 1 else [1, 2]
            self.assertEqual(irf.center, want)
            want = [2] if i is 1 else [3, 4]
            self.assertEqual(irf.width, want)
            want = [3] if i is 1 else [5, 6]
            self.assertEqual(irf.center_dispersion, want)
            want = [4] if i is 1 else [7, 8]
            self.assertEqual(irf.width_dispersion, want)
            want = [] if i is 1 else [9, 10]
            self.assertEqual(irf.scale, want)
            self.assertTrue(irf.normalize)

            if i is 1:
                self.assertTrue(irf.backsweep)
                self.assertEqual(irf.backsweep_period, 55)
            else:
                self.assertFalse(irf.backsweep)
                self.assertEqual(irf.backsweep_period, None)

            i = i + 1

    def test_k_matrices(self):
        self.assertTrue("km1" in self.model.k_matrices)
        self.assertTrue(np.array_equal(self.model.k_matrices["km1"]
                                       .asarray(),
                        np.array([[1, 3, 5, 7],
                                  [2, 0, 0, 0],
                                  [4, 0, 0, 0],
                                  [6, 0, 0, 0]]
                                 )
                                      )
                        )

    def test_shapes(self):

        self.assertTrue("shape1" in self.model.shapes)

        shape = self.model.shapes["shape1"]

        self.assertTrue(isinstance(shape, SpectralShapeGaussian))
        self.assertEqual(shape.amplitude, "shape.1")
        self.assertEqual(shape.location, "shape.2")
        self.assertEqual(shape.width, "shape.3")

    def test_megacomplexes(self):
        self.assertTrue(len(self.model.megacomplexes) is 3)

        i = 1
        for _ in self.model.megacomplexes:
            label = "cmplx{}".format(i)
            self.assertTrue(label in self.model.megacomplexes)
            megacomplex = self.model.megacomplexes[label]
            self.assertTrue(isinstance(megacomplex, KineticMegacomplex))
            self.assertTrue(megacomplex.label == label)
            self.assertEqual(megacomplex.k_matrices, ["km{}".format(i)])
            i = i + 1

    def test_parameter(self):
        allp = list(self.model.parameter.all_group())
        self.assertEqual(len(allp), 10)

        self.assertTrue(all(isinstance(p, Parameter) for p in allp))

        p = self.model.parameter.get('1')
        self.assertEqual(p.label, '1')
        self.assertEqual(p.value, 4.13E-02)
        self.assertEqual(p.min, 0)
        self.assertTrue(isinf(p.max))
        self.assertTrue(p.vary)

        for i in ['2', '3', '4', '5']:
            p = self.model.parameter.get(i)
            self.assertEqual(p.label, i)
            self.assertEqual(p.value, 1.0)
            self.assertFalse(p.vary)

        p = self.model.parameter.get('6')
        self.assertEqual(p.label, '6')
        self.assertEqual(p.value, 1.0)
        self.assertTrue(p.vary)

        p = self.model.parameter.get('7')
        self.assertEqual(p.label, '7')
        self.assertTrue(isnan(p.value))
        self.assertFalse(p.vary)

        p = self.model.parameter.get('spectral_equality')
        self.assertEqual(p.label, 'spectral_equality')
        self.assertEqual(p.value, 1.78)
        self.assertFalse(p.vary)
        self.assertTrue(p.fit)

        p = self.model.parameter.get_by_index(9)
        self.assertEqual(p.label, 'boundparam')
        self.assertEqual(p.value, 1.78)
        self.assertFalse(p.fit)
        self.assertTrue(p.vary)
        self.assertEqual(p.min, 0)
        self.assertEqual(p.max, 10)

        p = self.model.parameter.get_by_index(10)
        self.assertEqual(p.label, 'relatedparam')
        self.assertEqual(p.value, 1.78)
        self.assertFalse(p.vary)
        self.assertEqual(p.max, 2)
        self.assertEqual(p.expr, 'p_1 + 3')

        p = self.model.parameter.get('kinpar.k1')
        self.assertEqual(p.label, 'k1')
        self.assertEqual(p.value, 0.2)
        self.assertTrue(p.vary)

        p = self.model.parameter.get('kinpar.2')
        self.assertEqual(p.label, '2')
        self.assertEqual(p.value, 0.01)
        self.assertTrue(p.vary)

        p = self.model.parameter.get('kinpar.kf')
        self.assertEqual(p.label, 'kf')
        self.assertEqual(p.value, 0.0002)
        self.assertFalse(p.vary)

        p = self.model.parameter.get('shape.1')
        self.assertEqual(p.label, '1')
        self.assertEqual(p.value, 2.2)
        self.assertFalse(p.fit)
        self.assertTrue(p.vary)

        p = self.model.parameter.get('shape.rocks')
        self.assertEqual(p.label, 'rocks')
        self.assertEqual(p.value, 0.35)
        self.assertFalse(p.fit)
        self.assertTrue(p.vary)

        p = self.model.parameter.get('shape.myparam')
        self.assertEqual(p.label, 'myparam')
        self.assertEqual(p.value, 2.2)
        self.assertFalse(p.fit)
        self.assertTrue(p.vary)

        for i in range(3):

            p = self.model.parameter.get('testblock.{}'.format(i+1))
            self.assertEqual(p.min, 0)
            self.assertTrue(isinf(p.max))
