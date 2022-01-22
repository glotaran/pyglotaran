"""Deprecated functionality export for 'glotaran.examples.sequential'."""

sequential_deprecation_mapping = {
    "sim_model": "SIMULATION_MODEL",
    "dataset": "DATASET",
    "model": "MODEL",
    "scheme": "SCHEME",
}
shared_deprecation_mapping = {
    "wanted_parameter": "SIMULATION_PARAMETERS",
    "parameter": "PARAMETERS",
    "_time": "TIME_AXIS",
    "_spectral": "SPECTRAL_AXIS",
}


def __getattr__(attribute_name: str):
    from glotaran.deprecation.deprecation_utils import deprecate_module_attribute

    for deprecated, new in shared_deprecation_mapping.items():
        if attribute_name == deprecated:
            return deprecate_module_attribute(
                deprecated_qual_name=f"glotaran.examples.sequential.{deprecated}",
                new_qual_name=f"glotaran.testing.simulated_data.shared_decay.{new}",
                to_be_removed_in_version="0.8.0",
            )

    for deprecated, new in sequential_deprecation_mapping.items():
        if attribute_name == deprecated:
            return deprecate_module_attribute(
                deprecated_qual_name=f"glotaran.examples.sequential.{deprecated}",
                new_qual_name=f"glotaran.testing.simulated_data.sequential_spectral_decay.{new}",
                to_be_removed_in_version="0.8.0",
            )

    raise AttributeError(f"module {__name__} has no attribute {attribute_name}")
