from typing import TypeAlias
from typing import TypeVar

from attrs import define

from glotaran.parameter import Parameter

T = TypeVar("T")

ParameterType: TypeAlias = Parameter | str
ModelItemType: TypeAlias = T | str


def item(cls):
    return define(kw_only=True)(cls)


#  def has_label(cls) -> bool:
#      for field in fields(cls):
#          if field.name == "label":
#              return True
#      return False


class Item:
    pass
