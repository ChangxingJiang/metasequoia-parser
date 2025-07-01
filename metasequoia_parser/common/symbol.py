"""
符号对象
"""

import abc
import enum
from typing import Any, List, Type, cast

__all__ = [
    "Symbol",
    "Terminal",
    "End",
    "NonTerminal",
    "TerminalType",
]


class TerminalType(enum.IntEnum):
    """终结符类型的枚举类的样例"""

    @classmethod
    def create_by_terminal_name_list(cls,
                                     terminal_name_list: List[str],
                                     class_name: str = "TestTerminalType") -> Type["TerminalType"]:
        """使用终结符名称列表自动生成（仅限于测试）"""
        members = {"END": 0}
        for terminal_name in terminal_name_list:
            if terminal_name != "END":
                members[terminal_name] = len(members)
        return cast(Type[TerminalType], enum.IntEnum(class_name, members))  # 强制标注类型（动态生成子类因为无法扩展枚举类而失败）


class Symbol(abc.ABC):
    """符号对象（包括终结符和非终结符）"""

    def __init__(self, symbol_id: Any, value: Any):
        """符号对象的构造器

        Parameters
        ----------
        symbol_id : Any
            语义值（用于进行语义解析的值）
        value : Any
            实际值
        """
        self._symbol_id = symbol_id
        self._value = value

    @property
    def symbol_id(self) -> Any:
        """返回符号的语义值"""
        return self._symbol_id

    @property
    def symbol_value(self) -> Any:
        """返回符号的实际值"""
        return self._value

    @property
    def v(self) -> Any:
        """返回符号的实际值（等价于 symbol_value 的别名）"""
        return self._value

    @property
    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """如果符号为终结符则返回 True，否则返回 False"""

    @property
    def is_start(self) -> bool:
        """如果符号是开始符则返回 True，否则返回 False"""
        return False

    @property
    def is_end(self) -> bool:
        """如果符号是结束符则返回 True，否则返回 False"""
        return False

    def __eq__(self, other: "Symbol"):
        return self.symbol_id == other.symbol_id and self.symbol_value == other.symbol_value

    def __repr__(self) -> str:
        if isinstance(self._symbol_id, TerminalType):
            # 终结符的特殊展示方式
            return f"<{self.__class__.__name__} name={self._symbol_id.name}, value={repr(self._value)}>"
        return f"<{self.__class__.__name__} id={self._symbol_id}, value={self._value}>"


class Terminal(Symbol):
    """终结符"""

    @staticmethod
    def end() -> "Terminal":
        """获取结束符"""
        return End()

    @property
    def is_terminal(self) -> bool:
        """如果符号为终结符则返回 True，否则返回 False"""
        return True


class End(Terminal):
    """结束符"""

    def __init__(self):
        super().__init__(symbol_id=0, value=None)

    @property
    def is_end(self) -> bool:
        """如果符号是结束符则返回 True，否则返回 False"""
        return True

    def __repr__(self):
        return "<End id=0>"


class NonTerminal(Symbol):
    """非终结符"""

    @property
    def is_terminal(self) -> bool:
        """如果符号为终结符则返回 True，否则返回 False"""
        return False
