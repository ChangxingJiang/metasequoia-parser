"""
项目类
"""

import abc
import dataclasses
import enum
from typing import Callable, List, Optional, Tuple

from metasequoia_parser.common.grammar import CombineType

__all__ = [
    # 项目类型
    "ItemType",  # 项目类型的枚举类

    # 项目类
    "ItemBase",  # 文法项目的基类
    "Item0",  # 不提前查看下一个字符的文法项目：适用于 LR(0) 解析器和 SLR 解析器
    "ItemCentric",  # 项目核心：适用于 LALR(1) 解析器的同心项目集计算
    "Item1",  # 提前查看下一个字符的项目类：适用于 LR(1) 解析器和 LALR(1) 解析器
]


class ItemType(enum.Enum):
    """项目类型的枚举类"""

    INIT = 0  # 入口项目（Initial Item）：初始文法项目
    ACCEPT = 1  # 接收项目（Accept Item）：解析完成的文法项目

    SHIFT = 2  # 移进项目（Shift Item）：句柄位于开始或中间位置
    REDUCE = 3  # 规约项目（Reduce Item）：句柄位于结束位置，可以规约的项目


@dataclasses.dataclass(slots=True, frozen=True, eq=True, order=True)
class ItemBase(abc.ABC):
    # pylint: disable=R0902
    """项目类的抽象基类

    Attributes
    ----------
    nonterminal_id : str
        规约的非终结符名称（即所在语义组名称）
    before_handle : Tuple[str, ...]
        在句柄之前的符号名称的列表
    after_handle : Tuple[str, ...]
        在句柄之后的符号名称的列表
    item_type : ItemType
        项目类型
    action : Callable
        项目的规约行为函数
    successor_symbol : Optional[str]
        能够连接到后继项目的符号名称
    successor_item : Optional[ItemBase]
        连接到的后继项目对象
    """

    # 项目的基本信息（节点属性）
    nonterminal_id: int = dataclasses.field(kw_only=True)  # 规约的非终结符 ID（即所在语义组名称对应的 ID）
    before_handle: Tuple[int, ...] = dataclasses.field(kw_only=True)  # 在句柄之前的符号名称的列表
    after_handle: Tuple[int, ...] = dataclasses.field(kw_only=True)  # 在句柄之后的符号名称的列表
    item_type: ItemType = dataclasses.field(kw_only=True)  # 项目类型
    action: Callable = dataclasses.field(kw_only=True)  # 项目的规约行为函数

    # 项目的关联关系（节点出射边）
    successor_symbol: Optional[int] = dataclasses.field(kw_only=True)  # 能够连接到后继项目的符号名称
    successor_item: Optional["ItemBase"] = dataclasses.field(kw_only=True)  # 连接到的后继项目对象

    # 项目的 SR 优先级、结合方向和 RR 优先级
    sr_priority_idx: int = dataclasses.field(kw_only=True)  # 生成式的 SR 优先级序号（越大越优先）
    sr_combine_type: CombineType = dataclasses.field(kw_only=True)  # 生成式的 SR 合并顺序
    rr_priority_idx: int = dataclasses.field(kw_only=True)  # 生成式的 RR 优先级序号（越大越优先）

    @staticmethod
    @abc.abstractmethod
    def create(*args, **kwargs) -> "ItemBase":
        """项目对象的构造方法"""

    @abc.abstractmethod
    def __repr__(self) -> str:
        """将 ItemBase 转换为字符串表示"""


@dataclasses.dataclass(slots=True, frozen=True, eq=True, order=True)
class Item0(ItemBase):
    """不提前查看下一个字符的项目类：适用于 LR(0) 解析器和 SLR 解析器

    Attributes
    ----------
    successor_item : Optional[Item0]
        连接到的后继项目对象
    """

    successor_item: Optional["Item0"] = dataclasses.field(kw_only=True)  # 连接到的后继项目对象

    @staticmethod
    def create(reduce_name: int,
               before_handle: List[int],
               after_handle: List[int],
               action: Callable,
               item_type: ItemType,
               successor_symbol: Optional[int],
               successor_item: Optional["Item0"],
               sr_priority_idx: int,
               sr_combine_type: CombineType,
               rr_priority_idx: int
               ) -> "Item0":
        # pylint: disable=W0221
        # pylint: disable=R0913
        """项目对象的构造方法

        Parameters
        ----------
        reduce_name : str
            规约的非终结符名称（即所在语义组名称）
        before_handle : List[str]
            在句柄之前的符号名称的列表
        after_handle : List[str]
            在句柄之后的符号名称的列表
        item_type : ItemType
            项目类型
        action : Callable
            项目的规约行为函数
        successor_symbol : Optional[str]
            能够连接到后继项目的符号名称
        successor_item : Optional[Item0]
            连接到的后继项目对象
        sr_priority_idx : int
            生成式的 SR 优先级序号（越大越优先）
        sr_combine_type : CombineType
            生成式的 SR 合并顺序
        rr_priority_idx : int = dataclasses.field(kw_only=True)
            生成式的 RR 优先级序号（越大越优先）

        Returns
        -------
        Item0
            构造的 Item0 文法项目对象
        """
        return Item0(
            nonterminal_id=reduce_name,
            before_handle=tuple(before_handle),
            after_handle=tuple(after_handle),
            action=action,
            item_type=item_type,
            successor_symbol=successor_symbol,
            successor_item=successor_item,
            sr_priority_idx=sr_priority_idx,
            sr_combine_type=sr_combine_type,
            rr_priority_idx=rr_priority_idx
        )

    def __repr__(self) -> str:
        """将 ItemBase 转换为字符串表示"""
        before_symbol_str = " ".join(str(symbol) for symbol in self.before_handle)
        after_symbol_str = " ".join(str(symbol) for symbol in self.after_handle)
        return f"{self.nonterminal_id}->{before_symbol_str}·{after_symbol_str}"


@dataclasses.dataclass(slots=True, frozen=True, eq=True, order=True)
class ItemCentric:
    """项目核心：适用于 LALR(1) 解析器的同心项目集计算

    Parameters
    ----------
    reduce_name : str
        规约的非终结符名称（即所在语义组名称）
    before_handle : Tuple[str, ...]
        在句柄之前的符号名称的列表
    after_handle : Tuple[str, ...]
        在句柄之后的符号名称的列表
    """

    reduce_name: int = dataclasses.field(kw_only=True)  # 规约的非终结符名称（即所在语义组名称）
    before_handle: Tuple[int, ...] = dataclasses.field(kw_only=True)  # 在句柄之前的符号名称的列表
    after_handle: Tuple[int, ...] = dataclasses.field(kw_only=True)  # 在句柄之后的符号名称的列表

    def __repr__(self) -> str:
        """将 ItemBase 转换为字符串表示"""
        before_symbol_str = " ".join(str(symbol) for symbol in self.before_handle)
        after_symbol_str = " ".join(str(symbol) for symbol in self.after_handle)
        return f"{self.reduce_name}->{before_symbol_str}·{after_symbol_str}"


@dataclasses.dataclass(slots=True, frozen=True, eq=True, order=True)
class Item1(ItemBase):
    """提前查看下一个字符的项目类：适用于 LR(1) 解析器和 LALR(1) 解析器

    Attributes
    ----------
    successor_item : Optional[Item0]
        连接到的后继项目对象
    """

    successor_item: Optional["Item1"] = dataclasses.field(kw_only=True)  # 连接到的后继项目对象
    lookahead: int = dataclasses.field(kw_only=True)  # 展望符（终结符）

    @staticmethod
    def create(reduce_name: int,
               before_handle: List[int],
               after_handle: List[int],
               action: Callable,
               item_type: ItemType,
               successor_symbol: Optional[int],
               successor_item: Optional["Item1"],
               lookahead: int,
               sr_priority_idx: int,
               sr_combine_type: CombineType,
               rr_priority_idx: int
               ) -> "Item1":
        # pylint: disable=W0221
        # pylint: disable=R0913
        """项目对象的构造方法

        Parameters
        ----------
        reduce_name : str
            规约的非终结符名称（即所在语义组名称）
        before_handle : List[str]
            在句柄之前的符号名称的列表
        after_handle : List[str]
            在句柄之后的符号名称的列表
        item_type : ItemType
            项目类型
        action : Callable
            项目的规约行为函数
        successor_symbol : Optional[str]
            能够连接到后继项目的符号名称
        successor_item : Optional[Item1]
            连接到的后继项目对象
        lookahead : int
            展望符
        sr_priority_idx : int
            生成式的 SR 优先级序号（越大越优先）
        sr_combine_type : CombineType
            生成式的 SR 合并顺序
        rr_priority_idx : int = dataclasses.field(kw_only=True)
            生成式的 RR 优先级序号（越大越优先）

        Returns
        -------
        Item1
            构造的 Item1 文法项目对象
        """
        return Item1(
            nonterminal_id=reduce_name,
            before_handle=tuple(before_handle),
            after_handle=tuple(after_handle),
            action=action,
            item_type=item_type,
            successor_symbol=successor_symbol,
            successor_item=successor_item,
            lookahead=lookahead,
            sr_priority_idx=sr_priority_idx,
            sr_combine_type=sr_combine_type,
            rr_priority_idx=rr_priority_idx
        )

    @staticmethod
    def create_by_item0(item0: Item0, lookahead=lookahead) -> "Item1":
        """使用 Item0 构造 Item1

        Parameters
        ----------
        item0 : Item0
            构造的 Item0 文法项目对象
        lookahead : int
            展望符

        Returns
        -------
        Item1
            构造的 Item1 文法项目对象
        """
        if item0.successor_item is not None:
            successor_item1 = Item1.create_by_item0(item0.successor_item, lookahead)
        else:
            successor_item1 = item0.successor_item
        return Item1(
            nonterminal_id=item0.nonterminal_id,
            before_handle=item0.before_handle,
            after_handle=item0.after_handle,
            action=item0.action,
            item_type=item0.item_type,
            successor_symbol=item0.successor_symbol,
            successor_item=successor_item1,
            lookahead=lookahead,
            sr_priority_idx=item0.sr_priority_idx,
            sr_combine_type=item0.sr_combine_type,
            rr_priority_idx=item0.rr_priority_idx
        )

    def get_centric(self) -> ItemCentric:
        """获取项目核心：适用于 LALR(1) 解析器的同心项目集计算"""
        return ItemCentric(
            reduce_name=self.nonterminal_id,
            before_handle=self.before_handle,
            after_handle=self.after_handle,
        )

    def __repr__(self) -> str:
        """将 ItemBase 转换为字符串表示"""
        before_symbol_str = " ".join(str(symbol) for symbol in self.before_handle)
        after_symbol_str = " ".join(str(symbol) for symbol in self.after_handle)
        return f"{self.nonterminal_id}->{before_symbol_str}·{after_symbol_str},{self.lookahead}"
