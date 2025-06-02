"""
根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射，同时构造项目集闭包之间的关联关系
"""

import collections
from typing import Dict, List, Tuple

from metasequoia_parser.common import Grammar
from metasequoia_parser.common import Item0
from metasequoia_parser.common import Item1
from metasequoia_parser.common import Item1Set
from metasequoia_parser.functions.cal_nonterminal_all_start_terminal import cal_nonterminal_all_start_terminal
from metasequoia_parser.functions.closure_item1 import closure_item1
from metasequoia_parser.utils import LOGGER

__all__ = [
    "cal_core_to_item1_set_hash"
]


def cal_core_to_item1_set_hash(grammar: Grammar,
                               item0_list: List[Item0],
                               init_item0: Item0,
                               symbol_start_item0_list_hash: Dict[int, List[Item0]],
                               debug: bool = False
                               ) -> Dict[Tuple[Item1, ...], Item1Set]:
    # pylint: disable=R0914
    """根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射，同时构造项目集闭包之间的关联关系

    Parameters
    ----------
    grammar : Grammar
        语法类
    item0_list : List[Item0]
        所有 Item0 项目的列表
    init_item0 : Item0
        项目集闭包的核心项目（最高层级项目）
    symbol_start_item0_list_hash : Dict[int, List[Item0]]
        键为非终结符名称，值为非终结符对应开始项目的列表
    debug : bool, default = False
        是否打印调试模式日志

    Returns
    -------
    Dict[Tuple[Item1, ...], Item1Set]
        项目集核心项目元组到项目集闭包的映射
    """
    # 计算所有非终结符名称的列表
    nonterminal_name_list = list({item0.nonterminal_id for item0 in item0_list})

    # 计算每个非终结符中，所有可能的开头终结符
    nonterminal_all_start_terminal = cal_nonterminal_all_start_terminal(grammar, nonterminal_name_list)
    # print("nonterminal_all_start_terminal:", nonterminal_all_start_terminal)

    # 根据入口项的 LR(0) 项构造 LR(1) 项
    init_item1 = Item1.create_by_item0(init_item0, grammar.end_terminal)
    init_core_tuple = (init_item1,)

    # 初始化项目集闭包的广度优先搜索的队列：将入口项目集的核心项目元组添加到队列
    visited = {init_core_tuple}
    queue = collections.deque([init_core_tuple])

    # 初始化结果集（项目集核心项目元组到项目集闭包的映射）
    core_tuple_to_item1_set_hash = {}

    # 初始化项目集闭包之间的关联关系（采用个核心项目元组记录）
    item1_set_relation = []

    # 广度优先搜索遍历所有项目集闭包
    idx = 0
    while queue:
        core_tuple = queue.popleft()

        if debug is True:
            LOGGER.info(f"正在广度有限搜索遍历所有项目集闭包: 已处理={idx}, 队列中={len(queue)}")

        idx += 1

        # 根据项目集核心项目元组生成项目集闭包中包含的其他项目列表
        item1_list = closure_item1(grammar, core_tuple, symbol_start_item0_list_hash,
                                   nonterminal_all_start_terminal)

        # 构造项目集闭包并添加到结果集中
        item1_set = Item1Set.create(core_list=core_tuple, item_list=item1_list)
        core_tuple_to_item1_set_hash[core_tuple] = item1_set

        # 根据后继项目符号进行分组，计算出每个后继项目集闭包的核心项目元组
        successor_group = collections.defaultdict(set)
        for item1 in item1_set.all_item_list:
            if item1.successor_symbol is not None:
                successor_group[item1.successor_symbol].add(item1.successor_item)

        # 计算后继项目集的核心项目元组（排序以保证顺序稳定）
        successor_core_tuple_hash = {}
        for successor_symbol, sub_item1_set in successor_group.items():
            successor_core_tuple: Tuple[Item1, ...] = tuple(sorted(sub_item1_set, key=repr))
            successor_core_tuple_hash[successor_symbol] = successor_core_tuple

        # 记录项目集闭包之间的关联关系
        for successor_symbol, successor_core_tuple in successor_core_tuple_hash.items():
            item1_set_relation.append((core_tuple, successor_symbol, successor_core_tuple))

        # 将后继项目集闭包的核心项目元组添加到队列
        for successor_core_tuple in successor_core_tuple_hash.values():
            if successor_core_tuple not in visited:
                queue.append(successor_core_tuple)
                visited.add(successor_core_tuple)

    # print("len(visited):", len(visited))

    # 构造项目集之间的关系
    for from_core_tuple, successor_symbol, to_core_tuple in item1_set_relation:
        from_item1_set = core_tuple_to_item1_set_hash[from_core_tuple]
        to_item1_set = core_tuple_to_item1_set_hash[to_core_tuple]
        from_item1_set.set_successor(successor_symbol, to_item1_set)
        # print(from_item1_set.core_tuple, "->", to_item1_set.core_tuple)

    return core_tuple_to_item1_set_hash
