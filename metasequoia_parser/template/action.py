"""
模板行为函数
"""

from typing import Any, Callable

from metasequoia_parser.common.grammar import GrammarActionParams

__all__ = [
    "RETURN_0",
    "RETURN_1",
    "RETURN_NULL",
    "RETURN_EMPTY_LIST",
    "LIST_INIT_0",
    "LIST_APPEND_1",
    "LIST_APPEND_2",
    "RETURN_VALUE_1",
]

# 固定返回第 0 个元素Bison 的默认 Action 函数）
RETURN_0: Callable[[GrammarActionParams], Any] = lambda x: x[0]

# 固定返回地 1 个元素
RETURN_1: Callable[[GrammarActionParams], Any] = lambda x: x[0]

# 固定返回 True
RETURN_TRUE: Callable[[GrammarActionParams], Any] = lambda _: True

# 固定返回 False
RETURN_FALSE: Callable[[GrammarActionParams], Any] = lambda _: False

# 固定返回 None
RETURN_NULL: Callable[[GrammarActionParams], Any] = lambda _: None

# 固定返回 None
RETURN_EMPTY_LIST: Callable[[GrammarActionParams], Any] = lambda _: []

# 将第 0 个元素初始化为列表
LIST_INIT_0: Callable[[GrammarActionParams], Any] = lambda x: [x[0]]

# 将第 1 个元素 append 到第 0 个元素的列表上
LIST_APPEND_1: Callable[[GrammarActionParams], Any] = lambda x: x[0] + [x[1]]

# 将第 2 个元素 append 到第 0 个元素的列表上
LIST_APPEND_2: Callable[[GrammarActionParams], Any] = lambda x: x[0] + [x[2]]

# 将第 0 个元素的值初始化为 1
RETURN_VALUE_1: Callable[[GrammarActionParams], Any] = lambda _: 1
