"""
水杉语法解析器
"""

from metasequoia_parser import compiler
from metasequoia_parser import functions as _functions
from metasequoia_parser import lexical
from metasequoia_parser import parser
from metasequoia_parser import template
from metasequoia_parser.common import TerminalType
from metasequoia_parser.common import grammar
from metasequoia_parser.common import symbol

__all__ = [
    # 核心工具类
    "symbol",

    # 词法解析器的抽象基类
    "lexical",

    # 语法解析器及编译工具
    "compiler", "parser",

    # 语法定义工具
    "create_grammar", "create_group", "create_rule", "create_sr_priority", "template",
    "COMBINE_LEFT", "COMBINE_RIGHT", "COMBINE_NONASSOC",

    # 测试工具
    "TerminalType",
]

create_grammar = grammar.GrammarBuilder.create
create_group = grammar.GGroup.create
create_rule = grammar.GRule.create
create_sr_priority = grammar.SRPriority.create
COMBINE_LEFT = grammar.CombineType.LEFT
COMBINE_RIGHT = grammar.CombineType.RIGHT
COMBINE_NONASSOC = grammar.CombineType.NONASSOC

del grammar
