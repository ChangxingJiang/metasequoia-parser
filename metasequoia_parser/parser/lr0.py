"""
LR(0) 文法解析器
"""

from typing import Callable, Dict, List, Optional

from metasequoia_parser.common import ActionAccept, ActionGoto, ActionReduce, ActionShift
from metasequoia_parser.common import Grammar
from metasequoia_parser.common import Item0
from metasequoia_parser.common import Item0Set
from metasequoia_parser.common import ParserBase
from metasequoia_parser.functions import build_relation_between_item0_set
from metasequoia_parser.functions import cal_accept_item_from_item_list
from metasequoia_parser.functions import cal_all_item0_list
from metasequoia_parser.functions import cal_core_to_item0_set_hash
from metasequoia_parser.functions import cal_init_item_from_item_list
from metasequoia_parser.functions import cal_symbol_to_start_item_list_hash
from metasequoia_parser.functions import table_full_error


def create_lr_parsing_table_use_lr0(grammar: Grammar,
                                    core_to_status_hash: Dict[Item0, int],
                                    core_to_item0_set_hash: Dict[Item0, Item0Set],
                                    accept_item0_set: Item0Set
                                    ) -> List[List[Callable]]:
    # pylint: disable=R0801
    """构造 ACTION 二维表和 GOTO 二维表

    pylint: disable=R0801 -- 未提高相似算法的代码可读性，允许不同算法之间存在少量相同代码

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

    Returns
    -------
    table : List[List[Callable]]
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

        # 验证项目集闭包中的其他项目是否包含后继项目的状态是否与核心项目一致（仅 LR(0) 模型需要这个逻辑）
        has_successor = item0_set.core.successor_symbol is not None  # 核心项目是否包含项目集
        if has_successor is True:
            for sub_item0 in item0_set.item_list:
                assert sub_item0.successor_symbol is not None, f"项目集闭包 {item0_set} 的核心项目包含后继，但存在不包含后继的其他项目"
        else:
            for sub_item0 in item0_set.item_list:
                assert sub_item0.successor_symbol is None, f"项目集闭包 {item0_set} 的核心项目不包含后继，但存在包含后继的其他项目"

        # 获取项目集闭包对应的状态 ID
        status_id = core_to_status_hash[item0]

        if has_successor is True:
            # 如果包含后继项目，则遍历项目集闭包，填充 ACTION 表和 GOTO 表
            for sub_item0 in item0_set.all_item_list:
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
            reduce_action = ActionReduce(reduce_nonterminal_id=item0.nonterminal_id,
                                         n_param=len(item0.before_handle),
                                         reduce_function=item0.action)

            # 为所有终结符均添加 Reduce 行为
            for successor_idx in range(grammar.n_terminal):
                table[status_id][successor_idx] = reduce_action

    # 当入口项目集闭包接收到结束符时，填充 Accept 行为
    entrance_status_id = core_to_status_hash[accept_item0_set.core]
    table[entrance_status_id][grammar.end_terminal] = ActionAccept()

    # 将 ACTION 表和 GOTO 表中所有没有填充的位置全部置为 ERROR 行为（原地更新）
    table_full_error(table=table)

    return table


class ParserLR0(ParserBase):
    """LR(0) 解析器"""

    def create_action_table_and_goto_table(self):
        # pylint: disable=R0801
        """初始化 LR(0) 解析器

        1. 根据文法计算所有项目（Item0 对象），并生成项目之间的后继关系
        2. 根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表
        3. 从项目列表中获取入口项目
        4. 使用入口项目，广度优先搜索构造所有项目集闭包的列表（但不构造项目集闭包之间的关联关系）
        5. 创建 ItemSet 对象之间的关联关系（原地更新）
        6. 计算核心项目到项目集闭包 ID（状态）的映射表

        pylint: disable=R0801 -- 未提高相似算法的代码可读性，允许不同算法之间存在少量相同代码
        """
        # 根据文法计算所有项目（Item0 对象），并生成项目之间的后继关系
        item0_list = cal_all_item0_list(self.grammar)

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
        action_table = create_lr_parsing_table_use_lr0(
            grammar=self.grammar,
            core_to_status_hash=core_to_status_hash,
            core_to_item0_set_hash=core_to_item0_set_hash,
            accept_item0_set=accept_item0_set
        )

        return action_table, entrance_status
