"""
词法解析结果迭代器
"""

from typing import List

from metasequoia_parser.common.grammar import Grammar
from metasequoia_parser.common.symbol import Symbol, Terminal
from metasequoia_parser.lexical.lexical_base import LexicalBase

__all__ = [
    "LexicalIterator",
    "create_lexical_iterator_by_name_list",
]


class LexicalIterator(LexicalBase):
    """词法解析结果迭代器（主要用于测试场景）"""

    def __init__(self, lexical_data: List[Symbol]):
        self._lexical_data = lexical_data
        self._idx = 0

    def get_terminal(self) -> Symbol:
        """返回当前终结符"""
        if self._idx == len(self._lexical_data):
            return Terminal.end()
        return self._lexical_data[self._idx]

    def lex(self):
        res = self.get_terminal()
        self._idx += 1
        return res


def create_lexical_iterator_by_name_list(grammar: Grammar, name_list: List[str]):
    """【测试】使用终结符 name 列表构造词法解析结果迭代器，并令终结符的 value 与 name 一致"""
    return LexicalIterator([Terminal(symbol_id=grammar.terminal_type_enum[name], value=name) for name in name_list])
