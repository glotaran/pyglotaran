from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import megacomplex

#  from glotaran.parameter import Parameter
#  from glotaran.parameter import ParameterGroup


@megacomplex(dimension="global", properties={})
class SimpleTestMegacomplexGlobal(Megacomplex):
    def calculate_matrix(self, dataset_model, indices, **kwargs):
        axis = dataset_model.get_coordinates()
        assert "model" in axis
        assert "global" in axis
        axis = axis["global"]
        compartments = ["s1", "s2"]
        r_compartments = []
        array = np.zeros((axis.shape[0], len(compartments)))

        for i in range(len(compartments)):
            r_compartments.append(compartments[i])
            for j in range(axis.shape[0]):
                array[j, i] = (i + j) * axis[j]
        return xr.DataArray(array, coords=(("global", axis.data), ("clp_label", r_compartments)))

    def index_dependent(self, dataset_model):
        return False


@megacomplex(dimension="model", properties={"is_index_dependent": bool})
class SimpleTestMegacomplex(Megacomplex):
    def calculate_matrix(self, dataset_model, indices, **kwargs):
        axis = dataset_model.get_coordinates()
        assert "model" in axis
        assert "global" in axis

        axis = axis["model"]
        compartments = ["s1", "s2"]
        r_compartments = []
        array = np.zeros((axis.shape[0], len(compartments)))

        for i in range(len(compartments)):
            r_compartments.append(compartments[i])
            for j in range(axis.shape[0]):
                array[j, i] = (i + j) * axis[j]
        return xr.DataArray(array, coords=(("model", axis.data), ("clp_label", r_compartments)))

    def index_dependent(self, dataset_model):
        return self.is_index_dependent


class SimpleTestModel(Model):
    @classmethod
    def from_dict(cls, model_dict):
        return super().from_dict(
            model_dict,
            megacomplex_types={
                "model_complex": SimpleTestMegacomplex,
                "global_complex": SimpleTestMegacomplexGlobal,
            },
        )


#
#  @megacomplex(dimension="c", properties={"is_index_dependent": bool})
#  class SimpleKineticMegacomplex(Megacomplex):
#      def calculate_matrix(self, dataset_model, indices, **kwargs):
#          axis = dataset_model.get_data().coords
#          assert "c" in axis
#          assert "e" in axis
#          axis = axis["c"]
#          kinpar = -1 * np.asarray(dataset_model.kinetic)
#          if dataset_model.label == "dataset3":
#              # this case is for the ThreeDatasetDecay test
#              compartments = [f"s{i+2}" for i in range(len(kinpar))]
#          else:
#              compartments = [f"s{i+1}" for i in range(len(kinpar))]
#          array = np.exp(np.outer(axis, kinpar))
#          return xr.DataArray(array, coords=(("c", axis.data), ("clp_label", compartments)))
#
#      def index_dependent(self, dataset_model):
#          return self.is_index_dependent
#
#
#  def calculate_spectral_simple(dataset_descriptor, axis):
#      kinpar = -1 * np.array(dataset_descriptor.kinetic)
#      if dataset_descriptor.label == "dataset3":
#          # this case is for the ThreeDatasetDecay test
#          compartments = [f"s{i+2}" for i in range(len(kinpar))]
#      else:
#          compartments = [f"s{i+1}" for i in range(len(kinpar))]
#      array = np.asarray([[1 for _ in range(axis.size)] for _ in compartments])
#      return compartments, array.T
#
#
#  def calculate_spectral_gauss(dataset, axis):
#      location = np.asarray(dataset.location)
#      amp = np.asarray(dataset.amplitude)
#      delta = np.asarray(dataset.delta)
#
#      array = np.empty((location.size, axis.size), dtype=np.float64)
#
#      for i in range(location.size):
#          array[i, :] = amp[i]
# * np.exp(-np.log(2) * np.square(2 * (axis - location[i]) / delta[i]))
#      compartments = [f"s{i+1}" for i in range(location.size)]
#      return compartments, array.T
#
#
#
#
#
#  @model_attribute(
#      properties={
#          "kinetic": List[Parameter],
#      }
#  )
#  class DecayDatasetDescriptor(DatasetDescriptor):
#      pass
#
#
#  @model_attribute(
#      properties={
#          "kinetic": List[Parameter],
#          "location": {"type": List[Parameter], "allow_none": True},
#          "amplitude": {"type": List[Parameter], "allow_none": True},
#          "delta": {"type": List[Parameter], "allow_none": True},
#      }
#  )
#  class GaussianShapeDecayDatasetDescriptor(DatasetDescriptor):
#      pass
#
#
#  @model(
#      "one_channel",
#      attributes={},
#      dataset_type=DecayDatasetDescriptor,
#      model_dimension="c",
#      global_matrix=calculate_spectral_simple,
#      global_dimension="e",
#      megacomplex_types=SimpleKineticMegacomplex,
#      #  has_additional_penalty_function=lambda model: True,
#      #  additional_penalty_function=additional_penalty_typecheck,
#      #  has_matrix_constraints_function=lambda model: True,
#      #  constrain_matrix_function=constrain_matrix_function_typecheck,
#      #  retrieve_clp_function=retrieve_clp_typecheck,
#      grouped=lambda model: model.is_grouped,
#  )
#  class DecayModel(Model):
#      additional_penalty_function_called = False
#      constrain_matrix_function_called = False
#      retrieve_clp_function_called = False
#      is_grouped = False
#
#
#  @model(
#      "multi_channel",
#      attributes={},
#      dataset_type=GaussianShapeDecayDatasetDescriptor,
#      model_dimension="c",
#      global_matrix=calculate_spectral_gauss,
#      global_dimension="e",
#      megacomplex_types=SimpleKineticMegacomplex,
#      grouped=lambda model: model.is_grouped,
#      #  has_additional_penalty_function=lambda model: True,
#      #  additional_penalty_function=additional_penalty_typecheck,
#  )
#  class GaussianDecayModel(Model):
#      additional_penalty_function_called = False
#      constrain_matrix_function_called = False
#      retrieve_clp_function_called = False
#      is_grouped = False
#
#
#  class OneCompartmentDecay:
#      scale = 2
#      wanted_parameters = ParameterGroup.from_list([101e-4])
#      initial_parameters = ParameterGroup.from_list([100e-5, [scale, {"vary": False}]])
#
#      e_axis = np.asarray([1.0])
#      c_axis = np.arange(0, 150, 1.5)
#
#      model_dict = {
#          "megacomplex": {"m1": {"is_index_dependent": False}},
#          "dataset": {
#              "dataset1": {"initial_concentration": [], "megacomplex": ["m1"], "kinetic": ["1"]}
#          },
#      }
#      sim_model = DecayModel.from_dict(model_dict)
#      model_dict["dataset"]["dataset1"]["scale"] = "2"
#      model = DecayModel.from_dict(model_dict)
#
#
#  class TwoCompartmentDecay:
#      wanted_parameters = ParameterGroup.from_list([11e-4, 22e-5])
#      initial_parameters = ParameterGroup.from_list([10e-4, 20e-5])
#
#      e_axis = np.asarray([1.0])
#      c_axis = np.arange(0, 150, 1.5)
#
#      model = DecayModel.from_dict(
#          {
#              "megacomplex": {"m1": {"is_index_dependent": False}},
#              "dataset": {
#                  "dataset1": {
#                      "initial_concentration": [],
#                      "megacomplex": ["m1"],
#                      "kinetic": ["1", "2"],
#                  }
#              },
#          }
#      )
#      sim_model = model
#
#
#  class ThreeDatasetDecay:
#      wanted_parameters = ParameterGroup.from_list([101e-4, 201e-3])
#      initial_parameters = ParameterGroup.from_list([100e-5, 200e-3])
#
#      e_axis = np.asarray([1.0])
#      c_axis = np.arange(0, 150, 1.5)
#
#      e_axis2 = np.asarray([1.0, 2.01])
#      c_axis2 = np.arange(0, 100, 1.5)
#
#      e_axis3 = np.asarray([0.99, 3.0])
#      c_axis3 = np.arange(0, 150, 1.5)
#
#      model_dict = {
#          "megacomplex": {"m1": {"is_index_dependent": False}},
#          "dataset": {
#              "dataset1": {"initial_concentration": [], "megacomplex": ["m1"], "kinetic": ["1"]},
#              "dataset2": {
#                  "initial_concentration": [],
#                  "megacomplex": ["m1"],
#                  "kinetic": ["1", "2"],
#              },
#              "dataset3": {"initial_concentration": [], "megacomplex": ["m1"], "kinetic": ["2"]},
#          },
#      }
#      sim_model = DecayModel.from_dict(model_dict)
#      model = sim_model
#
#
#  class MultichannelMulticomponentDecay:
#      wanted_parameters = ParameterGroup.from_dict(
#          {
#              "k": [0.006, 0.003, 0.0003, 0.03],
#              "loc": [
#                  ["1", 14705],
#                  ["2", 13513],
#                  ["3", 14492],
#                  ["4", 14388],
#              ],
#              "amp": [
#                  ["1", 1],
#                  ["2", 2],
#                  ["3", 5],
#                  ["4", 20],
#              ],
#              "del": [
#                  ["1", 400],
#                  ["2", 100],
#                  ["3", 300],
#                  ["4", 200],
#              ],
#          }
#      )
#      initial_parameters = ParameterGroup.from_dict({"k": [0.006, 0.003, 0.0003, 0.03]})
#
#      e_axis = np.arange(12820, 15120, 50)
#      c_axis = np.arange(0, 150, 1.5)
#
#      sim_model = GaussianDecayModel.from_dict(
#          {
#              "compartment": ["s1", "s2", "s3", "s4"],
#              "megacomplex": {"m1": {"is_index_dependent": False}},
#              "dataset": {
#                  "dataset1": {
#                      "initial_concentration": [],
#                      "megacomplex": ["m1"],
#                      "kinetic": ["k.1", "k.2", "k.3", "k.4"],
#                      "location": ["loc.1", "loc.2", "loc.3", "loc.4"],
#                      "delta": ["del.1", "del.2", "del.3", "del.4"],
#                      "amplitude": ["amp.1", "amp.2", "amp.3", "amp.4"],
#                  }
#              },
#          }
#      )
#      model = GaussianDecayModel.from_dict(
#          {
#              "compartment": ["s1", "s2", "s3", "s4"],
#              "megacomplex": {"m1": {"is_index_dependent": False}},
#              "dataset": {
#                  "dataset1": {
#                      "initial_concentration": [],
#                      "megacomplex": ["m1"],
#                      "kinetic": ["k.1", "k.2", "k.3", "k.4"],
#                  }
#              },
#          }
#      )
