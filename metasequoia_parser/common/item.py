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
    """项目类的抽象基类"""

    @abc.abstractmethod
    def __repr__(self) -> str:
        """将 ItemBase 转换为字符串表示"""

    @abc.abstractmethod
    def get_symbol_id(self) -> int:
        """获取符号 ID"""

    @abc.abstractmethod
    def get_before_handle(self) -> Tuple[int, ...]:
        """获取句柄之前的符号名称的列表"""

    @abc.abstractmethod
    def get_after_handle(self) -> Tuple[int, ...]:
        """获取句柄之后的符号名称的列表"""

    @abc.abstractmethod
    def is_init(self) -> bool:
        """是否为入口项目"""

    @abc.abstractmethod
    def is_accept(self) -> bool:
        """是否为接受项目"""


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
class Item0(ItemBase):
    """不提前查看下一个字符的项目类：适用于 LR(0) 解析器和 SLR 解析器

    Attributes
    ----------
    successor_item : Optional[Item0]
        连接到的后继项目对象
    """

    # -------------------- 性能设计 --------------------
    # Item0 项目集唯一 ID
    # 通过在构造时添加 Item0 项目集的唯一 ID，从而将 Item0 项目集的哈希计算优化为直接获取唯一 ID
    id: int = dataclasses.field(kw_only=True, hash=True, compare=False)

    # 唯一 ID 生成计数器
    _INSTANCE_CNT = 0

    # -------------------- 项目的基本信息（节点属性）--------------------
    nonterminal_id: int = dataclasses.field(kw_only=True, hash=False, compare=True)  # 规约的非终结符 ID（即所在语义组名称对应的 ID）
    before_handle: Tuple[int, ...] = dataclasses.field(kw_only=True, hash=False, compare=True)  # 在句柄之前的符号名称的列表
    after_handle: Tuple[int, ...] = dataclasses.field(kw_only=True, hash=False, compare=True)  # 在句柄之后的符号名称的列表
    item_type: ItemType = dataclasses.field(kw_only=True, hash=False, compare=False)  # 项目类型
    action: Callable = dataclasses.field(kw_only=True, hash=False, compare=False)  # 项目的规约行为函数

    # -------------------- 项目的关联关系（节点出射边）--------------------
    # 能够连接到后继项目的符号名称（即 after_handle 中的第 1 个元素）
    successor_symbol: Optional[int] = dataclasses.field(kw_only=True, hash=False, compare=False)
    successor_item: Optional["Item0"] = dataclasses.field(kw_only=True, hash=False, compare=False)  # 连接到的后继项目对象

    # -------------------- 项目的 SR 优先级、结合方向和 RR 优先级 --------------------
    sr_priority_idx: int = dataclasses.field(kw_only=True, hash=False, compare=False)  # 生成式的 SR 优先级序号（越大越优先）
    sr_combine_type: CombineType = dataclasses.field(kw_only=True, hash=False, compare=False)  # 生成式的 SR 合并顺序
    rr_priority_idx: int = dataclasses.field(kw_only=True, hash=False, compare=False)  # 生成式的 RR 优先级序号（越大越优先）

    # 【性能设计】__repr__() 函数的返回值：
    # 之所以在初始化中指定，是因为这个对象是不可变的 dataclasses 类型，无法实现缓存器的逻辑
    repr_value: str = dataclasses.field(kw_only=True, hash=False, compare=False)

    # 【性能设计】项目集核心
    centric: ItemCentric = dataclasses.field(kw_only=True, hash=False, compare=False)

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
        # 【性能设计】提前计算 __repr__() 函数的返回值
        before_symbol_str = " ".join(str(symbol) for symbol in before_handle)
        after_symbol_str = " ".join(str(symbol) for symbol in after_handle)
        repr_value = f"{reduce_name}->{before_symbol_str}·{after_symbol_str}"

        # 【性能设计】Item0 项目集唯一 ID 计数器累加
        Item0._INSTANCE_CNT += 1

        # 【性能设计】提前计算项目集核心对象的返回值
        centric = ItemCentric(
            reduce_name=reduce_name,
            before_handle=tuple(before_handle),
            after_handle=tuple(after_handle),
        )

        return Item0(
            id=Item0._INSTANCE_CNT,
            nonterminal_id=reduce_name,
            before_handle=tuple(before_handle),
            after_handle=tuple(after_handle),
            action=action,
            item_type=item_type,
            successor_symbol=successor_symbol,
            successor_item=successor_item,
            sr_priority_idx=sr_priority_idx,
            sr_combine_type=sr_combine_type,
            rr_priority_idx=rr_priority_idx,
            repr_value=repr_value,
            centric=centric
        )

    def __repr__(self) -> str:
        """将 ItemBase 转换为字符串表示"""
        return self.repr_value

    def get_symbol_id(self) -> int:
        """获取符号 ID"""
        return self.nonterminal_id

    def get_before_handle(self) -> Tuple[int, ...]:
        """获取句柄之前的符号名称的列表"""
        return self.before_handle

    def get_after_handle(self) -> Tuple[int, ...]:
        """获取句柄之后的符号名称的列表"""
        return self.after_handle

    def is_init(self) -> bool:
        """是否为入口项目"""
        return self.item_type == ItemType.INIT

    def is_accept(self) -> bool:
        """是否为接收项目"""
        return self.item_type == ItemType.ACCEPT


@dataclasses.dataclass(slots=True, frozen=True, eq=True, order=True)
class Item1(ItemBase):
    """提前查看下一个字符的项目类：适用于 LR(1) 解析器和 LALR(1) 解析器

    Attributes
    ----------
    successor_item : Optional[Item0]
        连接到的后继项目对象
    """

    # -------------------- 性能设计 --------------------
    # 【性能设计】Item1 项目集唯一 ID
    # 通过在构造时添加 Item1 项目集的唯一 ID，从而将 Item1 项目集的哈希计算优化为直接获取唯一 ID
    id: int = dataclasses.field(kw_only=True, hash=True, compare=False)

    # 【性能设计】__repr__() 函数的返回值：
    # 之所以在初始化中指定，是因为这个对象是不可变的 dataclasses 类型，无法实现缓存器的逻辑
    repr_value: str = dataclasses.field(kw_only=True, hash=False, compare=False)

    # 【性能设计】享元模式缓存器
    # 通过享元模式，避免 Item1 对象被重复构造，以提高 Item1 对象的构造速度；通过这个字典也可以用于唯一 ID 的构造计数
    _INSTANCE_HASH = {}

    # -------------------- 项目的基本属性 --------------------
    item0: Item0 = dataclasses.field(kw_only=True, hash=False, compare=True)  # 连接到的后继项目对象
    lookahead: int = dataclasses.field(kw_only=True, hash=False, compare=True)  # 展望符（终结符）
    successor_item: Optional["Item1"] = dataclasses.field(kw_only=True, hash=False, compare=False)  # 连接到的后继项目对象

    @staticmethod
    def create_by_item0(item0: Item0, lookahead) -> "Item1":
        """采用享元模式，通过 Item0 对象和 lookahead 构造 Item1 对象

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
        item1 = Item1._INSTANCE_HASH.get((item0, lookahead))
        if item1 is not None:
            return item1

        successor_item1 = (Item1.create_by_item0(item0.successor_item, lookahead)
                           if item0.successor_item is not None else None)

        item1 = Item1(
            id=len(Item1._INSTANCE_HASH),
            item0=item0,
            successor_item=successor_item1,
            lookahead=lookahead,
            repr_value=f"{item0.repr_value},{lookahead}"  # 计算 __repr__ 函数的返回值
        )
        Item1._INSTANCE_HASH[(item0, lookahead)] = item1
        return item1

    def get_centric(self) -> ItemCentric:
        """获取项目核心：适用于 LALR(1) 解析器的同心项目集计算"""
        return self.item0.centric

    def __repr__(self) -> str:
        """将 ItemBase 转换为字符串表示"""
        return self.repr_value

    def get_symbol_id(self) -> int:
        """获取符号 ID"""
        return self.item0.nonterminal_id

    def get_before_handle(self) -> Tuple[int, ...]:
        """获取句柄之前的符号名称的列表"""
        return self.item0.before_handle

    def get_after_handle(self) -> Tuple[int, ...]:
        """获取句柄之后的符号名称的列表"""
        return self.item0.after_handle

    def is_init(self) -> bool:
        """是否为入口项目"""
        return self.item0.item_type == ItemType.INIT

    def is_accept(self) -> bool:
        """是否为接收项目"""
        return self.item0.item_type == ItemType.ACCEPT
