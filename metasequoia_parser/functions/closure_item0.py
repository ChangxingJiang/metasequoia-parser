"""
根据 Item0 生成项目集闭包（closure of item sets）中包含的项目列表
"""

import collections
from typing import Dict, List

from metasequoia_parser.common import Item0, ItemType

__all__ = [
    "closure_item0"
]


def closure_item0(item0: Item0, symbol_start_item0_list_hash: Dict[int, List[Item0]]) -> List[Item0]:
    """根据 Item0 生成项目集闭包（closure of item sets）中包含的项目列表

    Parameters
    ----------
    item0 : Item0
        项目集闭包的核心项目（最高层级项目）
    symbol_start_item0_list_hash : Dict[int, List[Item0]]
        键为非终结符名称，值为非终结符对应项目的列表

    Returns
    -------
    List[Item0]
        项目集闭包中包含的项目列表
    """
    if item0.item_type in {ItemType.ACCEPT, ItemType.REDUCE}:
        return []  # 如果项目组是规约项目，则不存在等价项目组，直接返回当前项目即为闭包

    # 初始化项目集闭包中包含的项目列表
    item_list = []

    # 获取核心项目句柄之后的第一个符号
    first_symbol = item0.after_handle[0]

    # 广度优先搜索所有的等价项目组
    visited_symbol_set = {first_symbol}  # 已访问过的句柄后第一个符号的集合
    queue = collections.deque([first_symbol])  # 待处理的句柄后第一个符号的集合
    while queue:
        symbol = queue.popleft()

        # 如果当前符号是终结符，则不存在等价项目
        if symbol not in symbol_start_item0_list_hash:
            continue

        # 添加生成 symbol 非终结符对应的所有项目，并将这些项目也加入继续寻找等价项目组的队列中
        for sub_item0 in symbol_start_item0_list_hash[symbol]:
            item_list.append(sub_item0)

            if len(sub_item0.after_handle) == 0:
                continue  # 跳过匹配 %empty 的项目

            new_symbol = sub_item0.after_handle[0]
            if new_symbol not in visited_symbol_set:
                visited_symbol_set.add(new_symbol)
                queue.append(new_symbol)

    return item_list
