"""
项目集闭包类
"""

import dataclasses
from typing import Dict, List, Tuple

from metasequoia_parser.common.item import Item0, Item1

__all__ = [
    "Item0Set",  # 不提前查看下一个字符的项目集闭包类：适用于 LR(0) 解析器和 SLR 解析器
    "Item1Set",  # 提前查看下一个字符的项目集闭包类：适用于 LR(1) 解析器和 LALR(1) 解析器
]


@dataclasses.dataclass(slots=True)
class Item0Set:
    """不提前查看下一个字符的项目集闭包类：适用于 LR(0) 解析器和 SLR 解析器"""

    core: Item0 = dataclasses.field(kw_only=True)  # 核心项目
    item_list: List[Item0] = dataclasses.field(kw_only=True)  # 项目集闭包中除核心项目外的其他等价项目
    successor_hash: Dict[int, "Item0Set"] = dataclasses.field(kw_only=True, default_factory=lambda: {})  # 后继项目的关联关系

    @staticmethod
    def create(core: Item0, item_list: List[Item0]) -> "Item0Set":
        """项目集闭包对象的构造方法"""
        return Item0Set(
            core=core,
            item_list=item_list
        )

    @property
    def all_item_list(self) -> List[Item0]:
        """返回项目集中的所有项目的列表，包括核心项目和其他等价项目；在输出的列表中，第 0 个项目为核心项目"""
        return [self.core] + self.item_list

    def set_successor(self, symbol: int, successor: "Item0Set"):
        """设置 symbol 对应的后继项目"""
        self.successor_hash[symbol] = successor

    def has_successor(self, symbol: int) -> bool:
        """如果存在 symbol 对应的后继项目则返回 True，否则返回 False"""
        return symbol in self.successor_hash

    def get_successor(self, symbol: int) -> "Item0Set":
        """获取 symbol 对象的后继项目"""
        return self.successor_hash[symbol]

    def __repr__(self):
        item_list_str = ",".join(str(item) for item in self.item_list)
        successors_str = ",".join(str(successor_idx) for successor_idx in self.successor_hash)
        return f"<core={self.core}, items={item_list_str}, successors=({successors_str})>"


@dataclasses.dataclass(slots=True)
class Item1Set:
    """提前查看下一个字符的项目集闭包类：适用于 LR(1) 解析器和 LALR(1) 解析器"""

    core_tuple: Tuple[Item1, ...] = dataclasses.field(kw_only=True)  # 核心项目
    item_list: List[Item1] = dataclasses.field(kw_only=True)  # 项目集闭包中除核心项目外的其他等价项目
    successor_hash: Dict[int, "Item1Set"] = dataclasses.field(kw_only=True, default_factory=lambda: {})  # 后继项目的关联关系

    @staticmethod
    def create(core_list: Tuple[Item1], item_list: List[Item1]) -> "Item1Set":
        """项目集闭包对象的构造方法"""
        return Item1Set(
            core_tuple=core_list,
            item_list=item_list
        )

    @property
    def all_item_list(self) -> List[Item1]:
        """返回项目集中的所有项目的列表，包括核心项目和其他等价项目；在输出的列表中，第 0 个项目为核心项目"""
        return list(self.core_tuple) + self.item_list

    def set_successor(self, symbol: int, successor: "Item1Set"):
        """设置 symbol 对应的后继项目"""
        self.successor_hash[symbol] = successor

    def has_successor(self, symbol: int) -> bool:
        """如果存在 symbol 对应的后继项目则返回 True，否则返回 False"""
        return symbol in self.successor_hash

    def get_successor(self, symbol: int) -> "Item1Set":
        """获取 symbol 对象的后继项目"""
        return self.successor_hash[symbol]

    def __repr__(self):
        core_tuple_str = "|".join(str(item) for item in self.core_tuple)
        return f"[{core_tuple_str}]"
