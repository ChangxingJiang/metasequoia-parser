"""
计算每个非终结符中，所有可能的开头终结符
"""

import collections
from typing import Dict, List, Set

from metasequoia_parser.common import Grammar

__all__ = [
    "cal_nonterminal_all_start_terminal"
]


def cal_nonterminal_all_start_terminal(grammar: Grammar, nonterminal_name_list: List[int]) -> Dict[int, Set[int]]:
    """计算每个非终结符中，所有可能的开头终结符

    Parameters
    ----------
    grammar : Grammar
        语法对象
    nonterminal_name_list : List[int]
        所有非终结符名称的列表

    Returns
    -------
    Dict[int, Set[int]]
        每个非终结标识符到其所有可能的开头终结符集合的映射
    """
    # 计算每个非终结符在各个生成式中的开头终结符和开头非终结符
    nonterminal_start_terminal = collections.defaultdict(set)  # "非终结符名称" 到其 "开头终结符的列表" 的映射
    nonterminal_start_nonterminal = collections.defaultdict(set)  # "非终结符名称" 到其 "开头非终结符的列表" 的映射
    for product in grammar.get_product_list():
        for symbol in product.symbol_id_list:
            if grammar.is_terminal(symbol):
                reduce_name = product.nonterminal_id
                nonterminal_start_terminal[reduce_name].add(symbol)
            else:
                reduce_name = product.nonterminal_id
                nonterminal_start_nonterminal[reduce_name].add(symbol)

            # 如果当前符号为终结符，或为不允许匹配 %empty 的非终结符，则说明后续符号已不可能再包含开头字符
            if not grammar.is_maybe_empty(symbol):
                break

    # 计算每个终结符直接或经过其他非终结符间接的开头终结符的列表
    nonterminal_all_start_terminal = collections.defaultdict(set)  # “非终结符名称” 到其 “直接或经由其他非终结符间接的开头终结符的列表” 的映射
    for nonterminal_name in nonterminal_name_list:
        # 广度优先搜索添加当前非终结符经由其他非终结符间接时间的开头终结符
        visited = {nonterminal_name}
        queue = collections.deque([nonterminal_name])
        while queue:
            now_nonterminal_name = queue.popleft()

            # 添加当前非终结符直接使用的开头终结符
            nonterminal_all_start_terminal[nonterminal_name] |= nonterminal_start_terminal[now_nonterminal_name]

            # 将当前非终结符使用的非终结符添加到队列
            for next_nonterminal_name in nonterminal_start_nonterminal[now_nonterminal_name]:
                if next_nonterminal_name not in visited:
                    queue.append(next_nonterminal_name)
                    visited.add(next_nonterminal_name)

    return nonterminal_all_start_terminal
