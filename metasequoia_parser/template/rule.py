"""
模板备选规则
"""

from metasequoia_parser.common.grammar import GRule
from metasequoia_parser.template import action

__all__ = [
    "EMPTY_RETURN_NULL",
    "EMPTY_RETURN_LIST",
]

# 不匹配任何符号，返回 None 的语义组（即 Bison 的 %empty）
EMPTY_RETURN_NULL = GRule.create(symbols=[], action=action.RETURN_NULL)

# 不匹配任何符号，返回空列表的语义组（即 Bison 的 %empty）
EMPTY_RETURN_LIST = GRule.create(symbols=[], action=action.RETURN_EMPTY_LIST)
