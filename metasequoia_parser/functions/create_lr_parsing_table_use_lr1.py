"""
使用 LR(1) 解析器的逻辑，构造 LR_Parsing_Table
"""

from typing import Callable, Dict, List, Optional, Tuple

from metasequoia_parser.common import ActionAccept, ActionGoto, ActionReduce, ActionShift
from metasequoia_parser.common import Grammar
from metasequoia_parser.common import Item1
from metasequoia_parser.common import Item1Set
from metasequoia_parser.exceptions import LR1GrammarError
from metasequoia_parser.functions.table_full_error import table_full_error

__all__ = [
    "create_lr_parsing_table_use_lr1"
]


def create_lr_parsing_table_use_lr1(grammar: Grammar,
                                    core_tuple_to_status_hash: Dict[Tuple[Item1, ...], int],
                                    core_tuple_to_item1_set_hash: Dict[Tuple[Item1, ...], Item1Set],
                                    accept_item1_set: Item1Set
                                    ) -> List[List[Callable]]:
    # pylint: disable=R0801
    """使用 LR(1) 解析器的逻辑，构造 LR_Parsing_Table

    pylint: disable=R0801 -- 未提高相似算法的代码可读性，允许不同算法之间存在少量相同代码

    Parameters
    ----------
    grammar : Grammar
        语法对象
    core_tuple_to_status_hash : Dict[Tuple[Item1, ...], int]
        核心项目到项目集闭包 ID（状态）的映射表
    core_tuple_to_item1_set_hash : Dict[Tuple[Item1, ...], Item1Set]
        核心项目到项目集闭包的映射（项目集闭包中包含项目列表，但不包含项目闭包之间关联关系）
    accept_item1_set : Item1Set
        接受项目集闭包

    Returns
    -------
    table : List[List[Callable]]
        ACTION 表 + GOTO 表
    """
    # 初始化 ACTION 二维表和 GOTO 二维表：第 1 维是状态 ID，第 2 维是符号 ID
    n_status = len(core_tuple_to_status_hash)
    table: List[List[Optional[Callable]]] = [[None] * grammar.n_symbol for _ in range(n_status)]

    # 遍历所有项目集闭包，填充 ACTION 表和 GOTO 表
    for core_tuple, item1_set in core_tuple_to_item1_set_hash.items():
        # 如果当前项目集闭包是接受项目集闭包，则不需要填充
        if item1_set.core_tuple == accept_item1_set.core_tuple:
            continue

        # 获取项目集闭包对应的状态 ID
        status_id = core_tuple_to_status_hash[core_tuple]

        # 根据项目集闭包的后继项目，填充 ACTION 表和 GOTO 表
        for successor_symbol, successor_item1_set in item1_set.successor_hash.items():
            next_status_id = core_tuple_to_status_hash[successor_item1_set.core_tuple]
            if grammar.is_terminal(successor_symbol):
                # 后继项目为终结符，填充 ACTION 表
                table[status_id][successor_symbol] = ActionShift(status=next_status_id)
            else:
                # 后继项目为非终结符，填充 GOTO 表
                table[status_id][successor_symbol] = ActionGoto(status=next_status_id)

        # 遍历不包含后继项目的项目，为 ACTION 表填充规约函数
        for sub_item1 in item1_set.all_item_list:
            if sub_item1.successor_symbol is None:
                reduce_action = ActionReduce(reduce_nonterminal_id=sub_item1.nonterminal_id,
                                             n_param=len(sub_item1.before_handle),
                                             reduce_function=sub_item1.action)

                # 根据展望符添加 Reduce 行为
                if table[status_id][sub_item1.lookahead] is not None:
                    raise LR1GrammarError(f"{item1_set} 项目集中的 {sub_item1} 项目，"
                                          f"后继项目 {sub_item1.lookahead} 冲突")

                table[status_id][sub_item1.lookahead] = reduce_action

    # 当入口项目集闭包接收到结束符时，填充 Accept 行为
    entrance_status_id = core_tuple_to_status_hash[accept_item1_set.core_tuple]
    table[entrance_status_id][grammar.end_terminal] = ActionAccept()

    # 将 ACTION 表和 GOTO 表中所有没有填充的位置全部置为 ERROR 行为（原地更新）
    table_full_error(table=table)

    return table
