"""
根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表
"""

import collections
from typing import Dict, List, NewType

from metasequoia_parser.common import ItemBase

__all__ = [
    "cal_symbol_to_start_item_list_hash"
]

# 定义 ItemBase 的子类作为泛型
T = NewType(name="T", tp=ItemBase)


def cal_symbol_to_start_item_list_hash(item_list: List[T]) -> Dict[int, List[T]]:
    """根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表

    Parameters
    ----------
    item_list : List[T]
        所有项目的列表（泛型 T 为 ItemBase 的子类）

    Returns
    -------
    Dict[int, List[T]]
        键为非终结符名称，值为非终结符对应项目的列表（泛型 T 为 ItemBase 的子类）
    """
    symbol_to_start_item_list_hash: Dict[int, List[T]] = collections.defaultdict(list)
    for item in item_list:
        if len(item.get_before_handle()) == 0:
            symbol_to_start_item_list_hash[item.get_symbol_id()].append(item)
    return symbol_to_start_item_list_hash
