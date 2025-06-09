"""
根据项目集核心项目元组（core_tuple）生成项目集闭包中包含的其他项目列表（item_list）
"""

import collections
from typing import Dict, List, Set, Tuple

from metasequoia_parser.common import Grammar
from metasequoia_parser.common import Item0
from metasequoia_parser.common import Item1
from metasequoia_parser.common import ItemType

__all__ = [
    "closure_item1"
]


def closure_item1(grammar: Grammar,
                  core_tuple: Tuple[Item1],
                  symbol_start_item0_list_hash: Dict[int, List[Item0]],
                  nonterminal_all_start_terminal: Dict[int, Set[int]]
                  ) -> List[Item1]:
    # pylint: disable=R0912
    # pylint: disable=R0914
    """根据项目集核心项目元组（core_tuple）生成项目集闭包中包含的其他项目列表（item_list）

    Parameters
    ----------
    grammar : Grammar
        语法类
    core_tuple : Tuple[Item1]
        项目集闭包的核心项目（最高层级项目）
    symbol_start_item0_list_hash : Dict[int, List[Item0]]
        键为非终结符名称，值为非终结符对应项目的列表
    nonterminal_all_start_terminal : Dict[int, Set[int]]
        每个非终结标识符到其所有可能的开头终结符集合的映射

    Returns
    -------
    List[Item1]
        项目集闭包中包含的项目列表
    """
    # 初始化项目集闭包中包含的其他项目列表
    item_set: Set[Item1] = set()

    # 初始化广度优先搜索的第 1 批节点
    visited_symbol_set = set()
    queue = collections.deque()
    for item1 in core_tuple:
        if item1.item_type in {ItemType.ACCEPT, ItemType.REDUCE}:
            continue  # 如果核心项是规约项目，则不存在等价项目组，跳过该项目即可

        # 将句柄之后的符号列表 + 展望符添加到队列中
        visited_symbol_set.add((item1.after_handle, item1.lookahead))
        queue.append((item1.after_handle, item1.lookahead))

    # 广度优先搜索所有的等价项目组
    while queue:
        after_handle, lookahead = queue.popleft()

        # 如果开头符号是终结符，则不存在等价项目
        first_symbol = after_handle[0]
        if grammar.is_terminal(first_symbol) is True:
            continue

        sub_item_set = set()  # 当前项目组之后的所有可能的 lookahead

        # 添加生成 symbol 非终结符对应的所有项目，并将这些项目也加入继续寻找等价项目组的队列中
        for item0 in symbol_start_item0_list_hash[first_symbol]:
            # 先从当前句柄后第 1 个元素向后继续遍历，添加自生型后继
            i = 1
            is_stop = False  # 是否已经找到不能匹配 %empty 的非终结符或终结符
            while i < len(after_handle):  # 向后逐个遍历符号，寻找展望符
                next_symbol = after_handle[i]

                # 如果遍历到的符号是终结符，则将该终结符添加为展望符，则标记 is_stop 并结束遍历
                if grammar.is_terminal(next_symbol):
                    sub_item_set.add(Item1.create_by_item0(item0, next_symbol))
                    is_stop = True
                    break

                # 如果遍历到的符号是非终结符，则遍历该非终结符的所有可能的开头终结符添加为展望符
                for start_terminal in nonterminal_all_start_terminal[next_symbol]:
                    sub_item_set.add(Item1.create_by_item0(item0, start_terminal))

                # 如果遍历到的非终结符不能匹配 %emtpy，则标记 is_stop 并结束遍历
                if not grammar.is_maybe_empty(next_symbol):
                    is_stop = True
                    break

                i += 1

            # 如果没有遍历到不能匹配 %empty 的非终结符或终结符，则添加继承型后继
            if is_stop is False:
                sub_item_set.add(Item1.create_by_item0(item0, lookahead))

        # if len(after_handle) == 1 and after_handle[0] == 7:
        #     print(f"after_handle: {after_handle}, lookahead: {lookahead}")

        # 将当前项目组匹配的等价项目组添加到所有等价项目组中
        item_set |= sub_item_set

        # 将等价项目组中需要继续寻找等价项目的添加到队列
        for sub_item1 in sub_item_set:
            if len(sub_item1.after_handle) == 0:
                continue  # 跳过匹配 %empty 的项目

            if (sub_item1.after_handle, sub_item1.lookahead) not in visited_symbol_set:
                visited_symbol_set.add((sub_item1.after_handle, sub_item1.lookahead))
                queue.append((sub_item1.after_handle, sub_item1.lookahead))

    return list(item_set)
