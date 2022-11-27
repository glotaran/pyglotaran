"""Deprecation helpers and place to put deprecated implementations till removing."""
from glotaran.deprecation.deprecation_utils import GlotaranApiDeprecationWarning
from glotaran.deprecation.deprecation_utils import GlotaranDeprecatedApiError
from glotaran.deprecation.deprecation_utils import deprecate
from glotaran.deprecation.deprecation_utils import deprecate_dict_entry
from glotaran.deprecation.deprecation_utils import deprecate_module_attribute
from glotaran.deprecation.deprecation_utils import deprecate_submodule
from glotaran.deprecation.deprecation_utils import raise_deprecation_error
from glotaran.deprecation.deprecation_utils import warn_deprecated

__all__ = [
    "deprecate",
    "deprecate_dict_entry",
    "deprecate_module_attribute",
    "deprecate_submodule",
    "raise_deprecation_error",
    "warn_deprecated",
    "GlotaranApiDeprecationWarning",
    "GlotaranDeprecatedApiError",
]
