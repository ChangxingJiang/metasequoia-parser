"""
解析器的各种异常类
"""

__all__ = [
    "ParserError",  # 解析器异常（各类解析异常的基类）
    "ParseError",  # 在解析器构造完成后，使用解析器解析的过程中抛出异常
    "GrammarError",  # 在构造解析器的过程中，文法异常，通常是因为出现了不支持的文法
    "LR0GrammarError",  # 使用了 LR(0) 解析器不支持的文法
    "SLRGrammarError",  # 使用了 SLR 解析器不支持的文法
    "LR1GrammarError",  # 使用了 LR(1) 解析器不支持的文法
    "LALR1GrammarError",  # 使用了 LALR(1) 解析器不支持的文法
]


class ParserError(Exception):
    """解析器异常（各类解析异常的基类）"""


class ParseError(ParserError):
    """在解析器构造完成后，使用解析器解析的过程中抛出异常"""


class GrammarError(ParserError):
    """在构造解析器的过程中，文法异常，通常是因为出现了不支持的文法"""


class LR0GrammarError(GrammarError):
    """使用了 LR(0) 解析器不支持的文法"""


class SLRGrammarError(GrammarError):
    """使用了 SLR 解析器不支持的文法"""


class LR1GrammarError(GrammarError):
    """使用了 LR(1) 解析器不支持的文法"""


class LALR1GrammarError(GrammarError):
    """使用了 LALR(1) 解析器不支持的文法"""
