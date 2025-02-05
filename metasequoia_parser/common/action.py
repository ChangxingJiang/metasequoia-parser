"""
行为类
"""

import abc
from typing import Callable

from metasequoia_parser.common.grammar import GrammarActionParams
from metasequoia_parser.common.parsing_context import ParsingContext
from metasequoia_parser.common.symbol import NonTerminal, Terminal

__all__ = [
    "ActionBase",
    "ActionShift",
    "ActionReduce",
    "ActionGoto",
    "ActionAccept",
    "ActionError",
]


class ActionBase(abc.ABC):
    # pylint: disable=R0903
    """行为类的抽象基类"""

    @abc.abstractmethod
    def __call__(self, context: ParsingContext, terminal: Terminal) -> int:
        """

        Parameters
        ----------
        context : ParsingContext
            状态管理器
        terminal : Terminal
            当前指针位置的终结符

        Returns
        -------
        int
            返回执行的行为类型：0=未知行为,1=移进(SHIFT)行为,2=规约(REDUCE)行为,3=GOTO行为,4=接收(ACCEPT)行为
        """


class ActionShift(ActionBase):
    """移进行为类"""

    def __init__(self, status: int):
        """

        Parameters
        ----------
        status : int
            需要压入栈中的状态
        """
        self._status = status

    @property
    def status(self) -> int:
        """返回需要压入栈中的状态"""
        return self._status

    def __call__(self, context: ParsingContext, terminal: Terminal) -> int:
        context.push(status=self._status, symbol=terminal)
        return 1

    def __repr__(self):
        return f"<Shift status={self._status}>"


class ActionReduce(ActionBase):
    """规约行为类"""

    def __init__(self, reduce_nonterminal_id: int, n_param: int, reduce_function: Callable):
        """

        Parameters
        ----------
        reduce_nonterminal_id : int
            规约生成的非终结符 ID
        n_param : int
            规约函数参数数量
        reduce_function : Callable
            规约函数可调用对象
        """
        self._reduce_name = reduce_nonterminal_id
        self._n_param = n_param
        self._reduce_function = reduce_function

    @property
    def reduce_name(self) -> int:
        """返回规约生成的非终结符 ID"""
        return self._reduce_name

    @property
    def n_param(self) -> int:
        """返回规约函数参数数量"""
        return self._n_param

    @property
    def reduce_function(self) -> Callable:
        """返回规约函数可调用对象"""
        return self._reduce_function

    def __call__(self, context: ParsingContext, terminal: Terminal) -> int:
        # 根据参数数量出栈参数
        symbols = []
        for _ in range(self._n_param):
            _, symbol = context.pop()
            symbols.append(symbol)
        symbols.reverse()

        # 调用规约函数生成规约后的符号
        reduce_value = self._reduce_function(GrammarActionParams(symbols))
        reduce_symbol = NonTerminal(symbol_id=self._reduce_name, value=reduce_value)

        # 将规约生成的符号添加到符号栈
        context.push_symbol(reduce_symbol)

        return 2

    def __repr__(self):
        return "<Reduce>"


class ActionGoto(ActionBase):
    """GOTO 行为类"""

    def __init__(self, status: int):
        """

        Parameters
        ----------
        status : int
            需要压入栈中的状态
        """
        self._status = status

    @property
    def status(self) -> int:
        """返回需要压入栈中的状态"""
        return self._status

    def __call__(self, context: ParsingContext, terminal: Terminal) -> int:
        context.push_status(status=self._status)
        return 3

    def __repr__(self):
        return f"<GOTO status={self._status}>"


class ActionAccept(ActionBase):
    """接受(ACCEPT)行为类"""

    def __call__(self, context: ParsingContext, terminal: Terminal) -> int:
        return 4

    def __repr__(self):
        return "<Accept>"


class ActionError(ActionBase):
    """错误(ERROR)行为类"""

    def __call__(self, context: ParsingContext, terminal: Terminal) -> int:
        return 5

    def __repr__(self):
        return "<Error>"
