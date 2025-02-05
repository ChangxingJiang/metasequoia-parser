"""
SLR 文法解析器
"""

import collections
from typing import Callable, Dict, List, Optional, Set

from metasequoia_parser.common import ActionAccept, ActionGoto, ActionReduce, ActionShift
from metasequoia_parser.common import Grammar
from metasequoia_parser.common import Item0, Item0Set
from metasequoia_parser.common.parser_base import ParserBase
from metasequoia_parser.exceptions import SLRGrammarError
from metasequoia_parser.functions import build_relation_between_item0_set
from metasequoia_parser.functions import cal_accept_item_from_item_list
from metasequoia_parser.functions import cal_all_item0_list
from metasequoia_parser.functions import cal_core_to_item0_set_hash
from metasequoia_parser.functions import cal_init_item_from_item_list
from metasequoia_parser.functions import cal_nonterminal_all_start_terminal
from metasequoia_parser.functions import cal_symbol_to_start_item_list_hash
from metasequoia_parser.functions import table_full_error


def cal_nonterminal_direct_follow_terminal(grammar: Grammar,
                                           nonterminal_all_start_terminal: Dict[int, Set[int]]
                                           ) -> Dict[int, Set[int]]:
    """计算每个非终结符的直接 FOLLOW 的终结符

    Parameters
    ----------
    grammar : Grammar
        语法对象
    nonterminal_all_start_terminal : Dict[int, Set[int]]
        每个非终结标识符到其所有可能的开头终结符集合的映射

    Returns
    -------
    Dict[int, Set[int]]
        每个非终结符的直接 FOLLOW 的终结符
    """
    nonterminal_direct_follow_terminal = collections.defaultdict(set)
    for product in grammar.get_product_list():
        symbols = product.symbol_id_list

        # 遍历生成式中的每个符号，计算它们的直接 FOLLOW 终结符
        for i, symbol_0 in enumerate(symbols):
            # 如果当前符号是终结符，则不查找它的直接 FOLLOW 终结符
            if grammar.is_terminal(symbol_0):
                continue

            # 从当前位置向后查找第 1 类直接 FOLLOW 终结符
            for j in range(i + 1, len(symbols)):
                symbol_1 = symbols[j]

                if grammar.is_terminal(symbol_1):
                    # 在目标非终结符后遇到终结符，则该终结符为目标非终结符的第 1 类直接 FOLLOW 的终结符，且之后不会存在其他 FOLLOW 的终结符
                    nonterminal_direct_follow_terminal[symbol_0].add(symbol_1)
                    break

                # 在目标非终结符后遇到非终结符，则该非终结符的所有可能的开头终结符均为目标非终结符的第 2 类 FOLLOW 的终结符
                nonterminal_direct_follow_terminal |= nonterminal_all_start_terminal[symbol_1]

                if not grammar.is_maybe_empty(symbol_1):
                    # 如果非终结符不匹配 %empty，则之后不会存在其他直接 FOLLOW 的终结符
                    break

    # 将结束符添加到入口符号作为直接 FOLLOW
    nonterminal_direct_follow_terminal[grammar.entrance_symbol].add(grammar.end_terminal)

    return nonterminal_direct_follow_terminal


def cal_nonterminal_use_end_nonterminal(grammar: Grammar) -> Dict[int, Set[int]]:
    """计算每个非终结符，在哪些其他非终结符的生成式中作为末尾位置出现

    Parameters
    ----------
    grammar : Grammar
        语法对象

    Returns
    -------
    Dict[int, Set[int]]
        非终结符名称到 “每个非终结符，在哪些其他非终结符的生成式中作为末尾位置出现” 的映射关系
    """
    nonterminal_use_end_nonterminal = collections.defaultdict(set)
    for product in grammar.get_product_list():
        symbols = product.symbol_id_list
        for i in range(len(symbols) - 1, -1, -1):
            symbol = symbols[i]
            # symbol_id = grammar.symbol_name_id_hash[symbol]

            # 如果遇到终结符，则之前不会再有作为结尾的非终结符
            if grammar.is_terminal(symbol):
                break

            # 当前 symbol 可能在目标非终结符的末尾位置出现
            nonterminal_use_end_nonterminal[symbol].add(product.nonterminal_id)

            # 如果非终结符不能匹配 %empty，则之前不会再有作为结尾的非终结符出现
            if not grammar.is_maybe_empty(symbol):
                break

    return nonterminal_use_end_nonterminal


def cal_follow_table(grammar: Grammar, item0_list: List[Item0]) -> Dict[int, Set[int]]:
    """根据 grammar 和 item0_list 计算每个符非终结符的 FOLLOW 表

    观察 FOLLOW 表生成逻辑，我们可以发现每个非终结符的 FOLLOW 有如下 4 种类型：
    1. 第 1 类直接 FOLLOW 的终结符
      1) T->aBd，其中 B 为非终结符，a、d 为终结符，则终结符 d 为非终结符 B 的第 1 类直接 FOLLOW 的终结符
      2) T->aBCd，其中 B 为非终结符，C 为可以匹配 %empty 的非终结符，a、d 为终结符，则终结符 d 为非终结符 B 的第 1 类直接 FOLLOW 的终结符
    2. 第 2 类直接 FOLLOW 的终结符
      1) T->aBC, C->d，其中 B、C 为非终结符，a、d 为终结符，则终结符 d为 非终结符 B 的第 2 类直接 FOLLOW 的终结符
      2) T->aBCD, C->%empty, D->e，其中 B、D 为非终结符，C 为可以匹配 %empty 的非终结符，a、e 为终结符，则终结符 e 为终结符 B 的第 2 类
      直接 FOLLOW 的终结符
    3. 间接 FOLLOW 的终结符
      1) B->cA, T->Bd，其中 A、B、T 为非终结符，c、d 为终结符，则终结符 d 为非终结符 A 的间接 FOLLOW 终结符
      2) B->cAE, T->Bd，其中 A、B、T 为非终结符，E 为可以匹配 %empty 的非终结符，a、c、d 为终结符，则终结符 d 为非终结符 A 的间接
       FOLLOW 终结符

    Parameters
    ----------
    grammar : Grammar
        语法对象
    item0_list : List[Item0]
        所有项目的列表

    Returns
    -------
    follow_table : Dict[str, Set[str]]
        每个非终结符的 FOLLOW 表（包括直接、间接 FOLLOW 的终结符），键为非终结符名称，值为 FOLLOW 的终结符的集合
    """
    # 计算所有非终结符名称的列表
    nonterminal_name_list = list({item0.nonterminal_id for item0 in item0_list})

    # 计算每个非终结符中，所有可能的开头终结符
    nonterminal_all_start_terminal = cal_nonterminal_all_start_terminal(grammar, nonterminal_name_list)

    # 计算每个非终结符的直接 FOLLOW 的终结符
    nonterminal_direct_follow_terminal = cal_nonterminal_direct_follow_terminal(grammar, nonterminal_all_start_terminal)

    # 计算每个非终结符，在哪些其他非终结符的生成式中作为末尾位置出现
    nonterminal_use_end_nonterminal = cal_nonterminal_use_end_nonterminal(grammar)

    # 计算每个非终结符直接和间接 FOLLOW 的终结符
    follow_table = collections.defaultdict(set)
    for nonterminal_name in nonterminal_name_list:
        # 广度优先搜索添加当前非终结符直接 FOLLOW 已经通过其他非终结符间接 FOLLOW 的终结符
        visited = {nonterminal_name}
        queue = collections.deque([nonterminal_name])
        while queue:
            next_nonterminal_name = queue.popleft()

            # 添加当前非终结符直接 FOLLOW 的终结符
            follow_table[nonterminal_name] |= nonterminal_direct_follow_terminal[next_nonterminal_name]

            # 根据计算每个非终结符，在哪些其他非终结符的生成式中作为末尾位置出现，并将这些非终结符添加到队列
            for next_nonterminal_name in nonterminal_use_end_nonterminal[next_nonterminal_name]:
                if next_nonterminal_name not in visited:
                    queue.append(next_nonterminal_name)
                    visited.add(next_nonterminal_name)

    return follow_table


def create_lr_parsing_table_use_slr(grammar: Grammar,
                                    core_to_status_hash: Dict[Item0, int],
                                    core_to_item0_set_hash: Dict[Item0, Item0Set],
                                    accept_item0_set: Item0Set,
                                    follow_table: Dict[int, Set[int]]
                                    ) -> List[List[Callable]]:
    # pylint: disable=R0801
    # pylint: disable=R0914
    """构造 ACTION 二维表和 GOTO 二维表

    Parameters
    ----------
    grammar : Grammar
        语法对象
    core_to_status_hash : Dict[Item0, int]
        核心项目到项目集闭包 ID（状态）的映射表
    core_to_item0_set_hash : Dict[Item0, Item0Set]
        核心项目到项目集闭包的映射（项目集闭包中包含项目列表，但不包含项目闭包之间关联关系）
    accept_item0_set : Item0Set
        接受项目集闭包
    follow_table : Dict[int, Set[int]]
        每个非终结符的 FOLLOW 表（包括直接、间接 FOLLOW 的终结符），键为非终结符名称，值为 FOLLOW 的终结符的集合

    Returns
    -------
    action_table : List[List[Callable]]
        ACTION 表 + GOTO 表
    """
    # 初始化 ACTION 二维表和 GOTO 二维表：第 1 维是状态 ID，第 2 维是符号 ID
    n_status = len(core_to_status_hash)
    table: List[List[Optional[Callable]]] = [[None] * grammar.n_symbol for _ in range(n_status)]

    # 遍历所有项目集闭包，填充 ACTION 表和 GOTO 表
    for item0, item0_set in core_to_item0_set_hash.items():
        # 如果当前项目集闭包是接受项目集闭包，则不需要填充
        if item0_set == accept_item0_set:
            continue

        # 获取项目集闭包对应的状态 ID
        status_id = core_to_status_hash[item0]

        # 如果包含后继项目，则遍历项目集闭包，填充 ACTION 表和 GOTO 表
        visited = set()
        for sub_item0 in item0_set.all_item_list:
            if sub_item0.successor_symbol is not None:
                if sub_item0.successor_symbol in visited:
                    raise SLRGrammarError(f"{item0.nonterminal_id} 的后继项目 {sub_item0.successor_symbol} 冲突")

                successor_idx = sub_item0.successor_symbol
                next_status_id = core_to_status_hash[sub_item0.successor_item]
                if grammar.is_terminal(successor_idx):
                    # 后继项目为终结符，填充 ACTION 表
                    table[status_id][successor_idx] = ActionShift(status=next_status_id)
                else:
                    # 后继项目为非终结符，填充 GOTO 表
                    table[status_id][successor_idx] = ActionGoto(status=next_status_id)
            else:
                # 如果不包含后继项目，则为 ACTION 表填充规约函数
                reduce_action = ActionReduce(reduce_nonterminal_id=sub_item0.nonterminal_id,
                                             n_param=len(sub_item0.before_handle),
                                             reduce_function=sub_item0.action)

                # 根据 FOLLOW 表添加 Reduce 行为
                for follow_id in follow_table[sub_item0.nonterminal_id]:
                    if sub_item0.successor_symbol in visited:
                        raise SLRGrammarError(f"{item0.nonterminal_id} 的后继项目 {follow_id} 冲突")
                    table[status_id][follow_id] = reduce_action

    # 当入口项目集闭包接收到结束符时，填充 Accept 行为
    entrance_status_id = core_to_status_hash[accept_item0_set.core]
    table[entrance_status_id][grammar.end_terminal] = ActionAccept()

    # 将 ACTION 表和 GOTO 表中所有没有填充的位置全部置为 ERROR 行为（原地更新）
    table_full_error(table=table)

    return table


class ParserSLR(ParserBase):
    """SLR 解析器"""

    def create_action_table_and_goto_table(self):
        # pylint: disable=R0801
        """初始化 SLR 解析器

        1. 根据文法计算所有项目（Item0 对象），并生成项目之间的后继关系
        2. 根据 grammar 和 item0_list 计算每个符非终结符的 FOLLOW 表
        3. 根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表
        4. 从项目列表中获取入口项目
        5. 使用入口项目，广度优先搜索构造所有项目集闭包的列表（但不构造项目集闭包之间的关联关系）
        6. 创建 ItemSet 对象之间的关联关系（原地更新）
        7. 计算核心项目到项目集闭包 ID（状态）的映射表

        pylint: disable=R0801 -- 未提高相似算法的代码可读性，允许不同算法之间存在少量相同代码
        """
        # 根据文法计算所有项目（Item0 对象），并生成项目之间的后继关系
        item0_list = cal_all_item0_list(self.grammar)

        # 根据 grammar 和 item0_list 计算每个符非终结符的 FOLLOW 表
        follow_table = cal_follow_table(self.grammar, item0_list)

        # 根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表
        symbol_to_start_item_list_hash = cal_symbol_to_start_item_list_hash(item0_list)

        # 从项目列表中获取入口项目
        init_item0 = cal_init_item_from_item_list(item0_list)

        # 使用入口项目，广度优先搜索构造所有项目集闭包的列表（但不构造项目集闭包之间的关联关系）
        core_to_item0_set_hash = cal_core_to_item0_set_hash(init_item0, symbol_to_start_item_list_hash)

        # 创建 ItemSet 对象之间的关联关系（原地更新）
        build_relation_between_item0_set(core_to_item0_set_hash)

        # 计算核心项目到项目集闭包 ID（状态）的映射表
        core_to_status_hash = {item0: i for i, item0 in enumerate(core_to_item0_set_hash.keys())}

        # 生成初始状态
        entrance_status = core_to_status_hash[init_item0]

        # 构造 ACTION 二维表
        accept_item0 = cal_accept_item_from_item_list(item0_list)
        accept_item0_set = core_to_item0_set_hash[accept_item0]
        action_table = create_lr_parsing_table_use_slr(
            grammar=self.grammar,
            core_to_status_hash=core_to_status_hash,
            core_to_item0_set_hash=core_to_item0_set_hash,
            accept_item0_set=accept_item0_set,
            follow_table=follow_table
        )

        return action_table, entrance_status
