"""
使用 LR(1) 解析器的逻辑，构造 LR_Parsing_Table
"""

import collections
from typing import Callable, Dict, List, Optional, Tuple

from metasequoia_parser.common import ActionAccept, ActionError, ActionGoto, ActionReduce, ActionShift
from metasequoia_parser.common import CombineType, Grammar
from metasequoia_parser.common import Item1
from metasequoia_parser.common import Item1Set
from metasequoia_parser.functions.table_full_error import table_full_error

__all__ = [
    "create_lr_parsing_table_use_lalr1"
]


def create_lr_parsing_table_use_lalr1(grammar: Grammar,
                                      core_tuple_to_status_hash: Dict[Tuple[Item1, ...], int],
                                      core_tuple_to_item1_set_hash: Dict[Tuple[Item1, ...], Item1Set],
                                      accept_item1_set: Item1Set
                                      ) -> List[List[Callable]]:
    # pylint: disable=R0801
    # pylint: disable=R0912
    # pylint: disable=R0914
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

    position_shift_hash = {}  # ACTION + GOTO 表位置到移进操作列表的哈希映射（每个位置至多有一个 Shift 行为）
    position_reduce_list_hash = collections.defaultdict(list)  # ACTION + GOTO 表位置到规约操作列表的哈希映射（每个位置可以有多个 Reduce 行为）

    # 遍历所有项目集闭包，填充 ACTION 表和 GOTO 表（当前项目集即使是接收项目集，也需要填充）
    for core_tuple, item1_set in core_tuple_to_item1_set_hash.items():
        # 获取项目集闭包对应的状态 ID
        status_id = core_tuple_to_status_hash[core_tuple]

        # 根据项目集闭包的后继项目，填充 ACTION 表和 GOTO 表
        for successor_symbol, successor_item1_set in item1_set.successor_hash.items():
            next_status_id = core_tuple_to_status_hash[successor_item1_set.core_tuple]
            if grammar.is_terminal(successor_symbol):
                # 后继项目为终结符，记录需要填充到 ACTION 表的 Shift 行为
                position_shift_hash[(status_id, successor_symbol)] = ActionShift(status=next_status_id)
            else:
                # 后继项目为非终结符，填充 GOTO 表
                table[status_id][successor_symbol] = ActionGoto(status=next_status_id)

        # 遍历不包含后继项目的项目，记录需要填充到 ACTION 表的 Reduce 行为
        for sub_item1 in item1_set.all_item_list:
            if sub_item1.item0.successor_symbol is None:
                reduce_action = ActionReduce(reduce_nonterminal_id=sub_item1.item0.nonterminal_id,
                                             n_param=len(sub_item1.item0.before_handle),
                                             reduce_function=sub_item1.item0.action)
                position_reduce_list_hash[(status_id, sub_item1.lookahead)].append((
                    sub_item1.item0.rr_priority_idx,  # RR 优先级
                    sub_item1.item0.sr_priority_idx,  # SR 优先级
                    sub_item1.item0.sr_combine_type,  # SR 合并顺序
                    reduce_action
                ))

    # ------------------------------ 处理 规约/规约冲突 ------------------------------
    position_reduce_hash = {}  # 解除 规约/规约冲突 后的每个位置的 Reduce 行为（至多有 1 个）
    for position, reduce_list in position_reduce_list_hash.items():
        reduce_list.sort(key=lambda x: x[0], reverse=True)  # 根据 RR 优先级倒序排序
        position_reduce_hash[position] = reduce_list[0]  # 选择 RR 优先级最大的 Reduce 行为

    # ------------------------------ 处理 移进/规约冲突 ------------------------------
    shift_position_set = set(position_shift_hash.keys())
    reduce_position_set = set(position_reduce_hash.keys())

    # 如果只有移进行为，没有移进/规约冲突，则直接写入移进行为
    for position in shift_position_set - reduce_position_set:
        status_id, successor_symbol = position
        action_shift = position_shift_hash[position]
        table[status_id][successor_symbol] = action_shift

    # 如果只有规约行为，没有移进/规约冲突，则直接写入规约行为
    for position in reduce_position_set - shift_position_set:
        status_id, successor_symbol = position
        _, _, _, action_reduce = position_reduce_hash[position]
        table[status_id][successor_symbol] = action_reduce

    # 如果既有移进行为、又有规约行为，存在移进/规约冲突，则进入处理逻辑
    for position in shift_position_set & reduce_position_set:
        status_id, successor_symbol = position

        # 获取移进行为信息
        action_shift = position_shift_hash[position]
        shift_sr_priority_idx = grammar.get_terminal_sr_priority_idx(successor_symbol)  # 移进行为 SR 优先级
        shift_sr_combine_type = grammar.get_terminal_sr_combine_type(successor_symbol)  # 移进行为 SR 结合顺序

        # 获取规约行为信息
        _, reduce_sr_priority_idx, _, action_reduce = position_reduce_hash[position]

        # print(f"【冲突处理】移进/规约冲突: "
        #       f"状态={status_id}, "
        #       f"下一个字符={grammar.get_symbol_name(successor_symbol)}({successor_symbol}), "
        #       f"移进行为优先级={shift_sr_priority_idx}, 规约行为优先级={reduce_sr_priority_idx}")

        if reduce_sr_priority_idx > shift_sr_priority_idx:
            # 如果要规约的规则的 SR 优先级高于下一个输入符号的 SR 优先级，则进行规约
            table[status_id][successor_symbol] = action_reduce
        elif reduce_sr_priority_idx < shift_sr_priority_idx:
            # 如果要规约的规则的 SR 优先级低于下一个输入符号的 SR 优先级，则进行移进
            table[status_id][successor_symbol] = action_shift
        else:  # reduce_sr_priority_idx == shift_sr_priority_idx
            # 如果要规约的规则的 SR 优先级与下一个输入符号的 SR 优先级一致，即均使用同一个终结符的 SR 优先级，则根据该符号的结合方向
            if shift_sr_combine_type == CombineType.LEFT:
                # 如果结合方向为从左到右，则进行规约
                table[status_id][successor_symbol] = action_reduce
            elif shift_sr_combine_type == CombineType.RIGHT:
                # 如果结合方向为从右到左，则进行移进
                table[status_id][successor_symbol] = action_shift
            else:
                # 如果既不是左结合也不是右结合，则抛出异常
                table[status_id][successor_symbol] = ActionError()

    # 当入口项目集闭包接收到结束符时，填充 Accept 行为
    entrance_status_id = core_tuple_to_status_hash[accept_item1_set.core_tuple]
    table[entrance_status_id][grammar.end_terminal] = ActionAccept()

    # 将 ACTION 表和 GOTO 表中所有没有填充的位置全部置为 ERROR 行为（原地更新）
    table_full_error(table=table)

    return table
