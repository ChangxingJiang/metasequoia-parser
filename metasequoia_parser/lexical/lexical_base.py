"""
语法解析器的接口类
"""

import abc
import dataclasses
from typing import List

from metasequoia_parser.common.symbol import Terminal

__all__ = [
    "LexicalState",
    "LexicalStateCached",
    "LexicalBase",
    "LexicalFSM",
    "END",
    "DEFAULT",
]

END = "<end>"  # 结束标识符
DEFAULT = "<default>"  # 默认标识符


@dataclasses.dataclass(slots=True)
class LexicalState:
    """词法解析器的缓存器的基类

    因为不同解析场景所需使用的状态各不相同，所以在基类中不设置状态属性，由子类实现
    """

    text: str = dataclasses.field(kw_only=True, init=True)  # 原始字符串
    pos_start: int = dataclasses.field(kw_only=True, default=0)  # 当前词语开始位置
    pos_now: int = dataclasses.field(kw_only=True, default=0)  # 当前位置

    def char(self) -> str:
        """返回当前字符"""
        if self.pos_now == len(self.text):
            return END
        return self.text[self.pos_now]

    def shift(self) -> None:
        """将当前字符添加到当前词语中，并将当前指针向后移动一个字符"""
        self.pos_now += 1

    def skip(self) -> None:
        """不将当前字符添加到当前词语中，并将当前指针向后移动一个字符"""
        raise KeyError("LexicalState 类不支持 skip 方法，请使用 LexicalStateCached 类")

    def end_word_exclude_now(self) -> None:
        """将 pos_now - 1 作为当前词语的最后一个字符，结束当前词语"""
        self.pos_start = self.pos_now

    def end_word_include_now(self) -> None:
        """将 pos_now 作为当前词语的最后一个字符，结束当前词语"""
        self.pos_now += 1
        self.pos_start = self.pos_now

    def end_word_exclude_now_and_return(self) -> str:
        """将 pos_now - 1 作为当前词语的最后一个字符，结束当前词语，并返回当前词语"""
        word = self.text[self.pos_start:self.pos_now]
        self.pos_start = self.pos_now
        return word

    def end_word_include_now_and_return(self) -> str:
        """将 pos_now 作为当前词语的最后一个字符，结束当前词语，并返回当前词语"""
        self.pos_now += 1
        word = self.text[self.pos_start:self.pos_now]
        self.pos_start = self.pos_now
        return word


@dataclasses.dataclass(slots=True)
class LexicalStateCached(LexicalState):
    """词法解析器的缓存器：通过每个字符缓存当前词语，而不是存储开始位置和结束位置，从而支持需要在词法解析时处理转义符的场景"""

    cache: List[str] = dataclasses.field(kw_only=True, default_factory=lambda: [])

    def add(self, char: str) -> None:
        """将额外的字符 char 添加到当前词语中"""
        self.cache.append(char)

    def shift(self) -> None:
        """将当前字符添加到当前词语中，并将当前指针向后移动一个字符"""
        self.cache.append(self.char())
        self.pos_now += 1

    def skip(self) -> None:
        """不将当前字符添加到当前词语中，并将当前指针向后移动一个字符"""
        self.pos_now += 1

    def end_word_exclude_now(self) -> None:
        """将 pos_now - 1 作为当前词语的最后一个字符，结束当前词语"""
        self.cache = []

    def end_word_include_now(self) -> None:
        """将 pos_now 作为当前词语的最后一个字符，结束当前词语"""
        self.pos_now += 1
        self.cache = []

    def end_word_exclude_now_and_return(self) -> str:
        """将 pos_now - 1 作为当前词语的最后一个字符，结束当前词语，并返回当前词语"""
        word = "".join(self.cache)
        self.cache = []
        self.pos_start = self.pos_now
        return word

    def end_word_include_now_and_return(self) -> str:
        """将 pos_now 作为当前词语的最后一个字符，结束当前词语，并返回当前词语"""
        self.cache.append(self.char())
        self.pos_now += 1
        word = "".join(self.cache)
        self.cache = []
        self.pos_start = self.pos_now
        return word


class LexicalBase(abc.ABC):
    # pylint: disable=R0903
    """词法解析结果的抽象基类"""

    @abc.abstractmethod
    def lex(self) -> Terminal:
        """获取当前终结符"""


class LexicalFSM(LexicalBase):
    # pylint: disable=R0903
    """词法解析器自动机的抽象基类"""

    def __init__(self, text: str):
        self.text: str = text

    @abc.abstractmethod
    def lex(self) -> Terminal:
        """解析并生成一个终结符"""
