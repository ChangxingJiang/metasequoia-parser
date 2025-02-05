"""
LR(1) 文法解析器
"""

from metasequoia_parser.common import Item1
from metasequoia_parser.common import ParserBase
from metasequoia_parser.functions import cal_accept_item_from_item_list
from metasequoia_parser.functions import cal_all_item0_list
from metasequoia_parser.functions import cal_core_to_item1_set_hash
from metasequoia_parser.functions import cal_init_item_from_item_list
from metasequoia_parser.functions import cal_symbol_to_start_item_list_hash
from metasequoia_parser.functions import create_lr_parsing_table_use_lr1


class ParserLR1(ParserBase):
    """LR(1) 解析器"""

    def create_action_table_and_goto_table(self):
        # pylint: disable=R0801
        """初始化 LR(1) 解析器

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
        core_tuple_to_item1_set_hash = cal_core_to_item1_set_hash(self.grammar, item0_list, init_item0,
                                                                  symbol_to_start_item_list_hash)

        # 计算核心项目到项目集闭包 ID（状态）的映射表
        core_tuple_to_status_hash = {core_tuple: i for i, core_tuple in enumerate(core_tuple_to_item1_set_hash.keys())}

        # 生成初始状态
        init_item1 = Item1.create_by_item0(init_item0, self.grammar.end_terminal)
        entrance_status = core_tuple_to_status_hash[(init_item1,)]

        # 构造 ACTION 表 + GOTO 表
        accept_item0 = cal_accept_item_from_item_list(item0_list)
        accept_item1 = Item1.create_by_item0(accept_item0, self.grammar.end_terminal)
        accept_item1_set = core_tuple_to_item1_set_hash[(accept_item1,)]

        table = create_lr_parsing_table_use_lr1(
            grammar=self.grammar,
            core_tuple_to_status_hash=core_tuple_to_status_hash,
            core_tuple_to_item1_set_hash=core_tuple_to_item1_set_hash,
            accept_item1_set=accept_item1_set
        )

        return table, entrance_status
