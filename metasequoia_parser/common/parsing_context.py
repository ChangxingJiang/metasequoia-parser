"""
语法解析过程中的状态管理器
"""

import enum
from typing import List, Tuple

from metasequoia_parser.common.grammar import Grammar
from metasequoia_parser.common.symbol import Symbol

__all__ = [
    "ParsingContext"
]


class ParsingContext:
    """状态管理器

    Attributes
    ----------
    _status_stack : List[int]
        状态栈
    _symbol_stack : List[Symbol]
        符号栈
    """

    def __init__(self):
        self._status_stack: List[int] = []  # 状态栈
        self._symbol_stack: List[Symbol] = []  # 符号栈

    def push(self, status: int, symbol: Symbol) -> None:
        """入栈一个元素（状态和符号）

        Parameters
        ----------
        status : int
            状态
        symbol : Symbol
            符号（终结符或非终结符均可）
        """
        self._status_stack.append(status)
        self._symbol_stack.append(symbol)

    def push_status(self, status: int):
        """入栈一个状态（在 GOTO 时调用，使符号栈和状态栈重新平衡）

        Parameters
        ----------
        status : int
            状态
        """
        self._status_stack.append(status)

    def push_symbol(self, symbol: Symbol):
        """入栈一个符号（在规约时调用，令符号栈比状态栈中多一个元素，等待 GOTO 时调用 push_status 重新平衡）

        Parameters
        ----------
        symbol : Symbol
            符号（终结符或非终结符均可）
        """
        self._symbol_stack.append(symbol)

    def pop(self) -> Tuple[int, Symbol]:
        """出栈一个元素（状态和符号）

        Returns
        -------
        status : int
            状态
        symbol : Symbol
            符号（终结符或非终结符均可）
        """
        status = self._status_stack.pop()
        symbol = self._symbol_stack.pop()
        return status, symbol

    def top(self) -> Tuple[int, Symbol]:
        """查看栈顶的状态和符号

        Returns
        -------
        status : int
            状态
        symbol : Symbol
            符号（终结符或非终结符均可）
        """
        return self._status_stack[-1], self._symbol_stack[-1]

    def top_status(self) -> int:
        """查看栈顶元素的状态

        Returns
        -------
        status : int
            状态
        """
        return self._status_stack[-1]

    def top_symbol(self) -> Symbol:
        """查看栈顶元素的符号

        Returns
        -------
        symbol : Symbol
            符号（终结符或非终结符均可）
        """
        return self._symbol_stack[-1]

    def __repr__(self):
        return f"<ParsingContext status_stack={self._status_stack}, symbol_stack={self._symbol_stack}>"

    def print(self, grammar: Grammar) -> str:
        """返回解析器上下文的打印信息字符串"""
        symbol_stack = []
        for symbol in self._symbol_stack:
            if isinstance(symbol, enum.IntEnum):
                # 终结符的特殊展示方式
                symbol_stack.append(f"<name={symbol.name}, "
                                    f"value={repr(symbol.value)}>")
            else:
                symbol_stack.append(f"<name={grammar.get_symbol_name(symbol.symbol_id)}, "
                                    f"value={symbol.symbol_value}>")
        symbol_stack_str = "; ".join(symbol_stack)
        return f"【解析器上下文】状态栈={self._status_stack}, 符号栈={symbol_stack_str}"
