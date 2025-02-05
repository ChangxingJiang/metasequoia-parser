"""
根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射但不构造项目集闭包之间的关联关系）
"""

import collections
from typing import Dict, List

from metasequoia_parser.common import Item0
from metasequoia_parser.common import Item0Set
from metasequoia_parser.functions.closure_item0 import closure_item0

__all__ = [
    "cal_core_to_item0_set_hash",
]


def cal_core_to_item0_set_hash(init_item0: Item0,
                               symbol_start_item0_list_hash: Dict[int, List[Item0]]) -> Dict[Item0, Item0Set]:
    """根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射但不构造项目集闭包之间的关联关系）

    Parameters
    ----------
    init_item0 : Item0
        项目集闭包的核心项目（最高层级项目）
    symbol_start_item0_list_hash : Dict[int, List[Item0]]
        键为非终结符名称，值为非终结符对应开始项目的列表

    Returns
    -------
    Dict[Item0, Item0Set]
        核心项目到项目集闭包的映射（项目集闭包中包含项目列表，但不包含项目闭包之间关联关系）
    """
    # 将入口项目添加到广度优先搜索的队列中
    visited = {init_item0}
    queue = collections.deque([init_item0])

    # 初始化结果集
    core_to_item0_set_hash = {}

    # 广度优先搜索遍历所有项目集闭包
    while queue:
        item0 = queue.popleft()

        # 根据 Item 生成项目集闭包中包含的项目列表
        item0_list = closure_item0(item0, symbol_start_item0_list_hash)

        # 构造项目集闭包并添加到结果集中
        item0_set = Item0Set.create(core=item0, item_list=item0_list)
        core_to_item0_set_hash[item0] = item0_set

        # 获取每个项目的作为后继项目集闭包的核心项目
        for sub_item0 in item0_set.all_item_list:
            successor_item = sub_item0.successor_item

            if successor_item is None:
                continue  # 没有后继项目
            if successor_item in visited:
                continue  # 后继项目已经处理

            queue.append(successor_item)
            visited.add(successor_item)

    return core_to_item0_set_hash
