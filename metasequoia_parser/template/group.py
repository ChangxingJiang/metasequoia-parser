"""
模板备选规则
"""
import enum
from typing import Any, Optional, Union

from metasequoia_parser.common.grammar import GGroup, GRule
from metasequoia_parser.template import action

__all__ = [
    "EMPTY_NULL",
    "EMPTY_LIST",
    "create_opt_group",
    "create_multi_group",
]

# 不匹配的语义组（即 Bison 的 %empty）
EMPTY_NULL = GRule.create(symbols=[], action=action.RETURN_NULL)

# 不匹配的语义组（即 Bison 的 %empty）
EMPTY_LIST = GRule.create(symbols=[], action=action.RETURN_EMPTY_LIST)


def create_opt_group(opt_name: str, group: Union[str, enum.IntEnum], empty_default: Optional[Any] = None) -> GGroup:
    """构造名称为 opt_name 语义组，该语义组用于匹配可选的 name 语义组

    Parameters
    ----------
    opt_name : str
        构造的语义组名称
    group : Union[str, enum.IntEnum]
        原始语义组
    empty_default : Optional[Any], default = None
        没有匹配时的返回值
    """
    return GGroup.create(
        name=opt_name,
        rules=[
            GRule.create(symbols=[group]),
            GRule.create(symbols=[], action=lambda _: empty_default)
        ]
    )


def create_multi_group(multi_name: str, group: Union[str, enum.IntEnum], sep: Optional[enum.IntEnum] = None) -> GGroup:
    """构造名称为 multi_name 语义组，该语义组用于匹配任意数量，使用 sep 分隔的 name 语义组

    Parameters
    ----------
    multi_name : str
        构造的语义组名称
    group : Union[str, enum.IntEnum]
        原始语义组
    sep : Optional[Terminal], default = None
        分隔符的终结符
    """
    if sep is None:
        return GGroup.create(
            name=multi_name,
            rules=[
                GRule.create(symbols=[multi_name, group], action=action.LIST_APPEND_1),
                GRule.create(symbols=[group], action=action.LIST_INIT_0),
            ]
        )

    return GGroup.create(
        name=multi_name,
        rules=[
            GRule.create(symbols=[multi_name, sep, group], action=action.LIST_APPEND_2),
            GRule.create(symbols=[group], action=action.LIST_INIT_0),
        ]
    )
