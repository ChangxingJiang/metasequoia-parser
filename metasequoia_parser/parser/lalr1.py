"""
LR(1) 文法解析器
"""

import collections
from typing import Dict, List, Set, Tuple

from metasequoia_parser.common import Grammar
from metasequoia_parser.common import Item1
from metasequoia_parser.common import Item1Set
from metasequoia_parser.common import ItemCentric
from metasequoia_parser.common import ParserBase
from metasequoia_parser.functions import cal_accept_item_from_item_list
from metasequoia_parser.functions import cal_all_item0_list
from metasequoia_parser.functions import cal_core_to_item1_set_hash
from metasequoia_parser.functions import cal_init_item_from_item_list
from metasequoia_parser.functions import cal_symbol_to_start_item_list_hash
from metasequoia_parser.functions import create_lr_parsing_table_use_lalr1
from metasequoia_parser.utils import LOGGER


def cal_core_tuple_to_before_item1_set_hash(core_tuple_to_item1_set_hash: Dict[Tuple[Item1, ...], Item1Set]
                                            ) -> Dict[Tuple[Item1, ...], List[Tuple[int, Item1Set]]]:
    """计算核心项目元组到该项目集的前置项目集的映射表

    Parameters
    ----------
    core_tuple_to_item1_set_hash : Dict[Tuple[Item1, ...], List[Item1Set]]
        项目集核心项目元组到项目集闭包的映射

    Returns
    -------
    Dict[Tuple[Item1, ...], List[Tuple[int, Item1Set]]]
        核心项目元组到该项目集的前置项目集的映射表
    """
    core_tuple_to_before_item1_set_hash = collections.defaultdict(list)
    for _, item1_set in core_tuple_to_item1_set_hash.items():
        for successor_symbol, successor_item1_set in item1_set.successor_hash.items():
            core_tuple_to_before_item1_set_hash[successor_item1_set.core_tuple].append((successor_symbol, item1_set))
    return core_tuple_to_before_item1_set_hash


def cal_concentric_hash(core_tuple_to_item1_set_hash: Dict[Tuple[Item1, ...], Item1Set]
                        ) -> Dict[Tuple[ItemCentric, ...], List[Item1Set]]:
    """计算项目集核心，并根据项目集的核心（仅包含规约符、符号列表和句柄的核心项目元组）进行聚合

    Parameters
    ----------
    core_tuple_to_item1_set_hash : Dict[Tuple[Item1, ...], Item1Set]
        项目集核心项目元组到项目集闭包的映射

    Returns
    -------
    Dict[Tuple[ItemCentric, ...], List[Item1Set]]
        根据项目集核心聚合后的项目集
    """
    concentric_hash = collections.defaultdict(list)
    for core_tuple, item1_set in core_tuple_to_item1_set_hash.items():
        # 计算项目集核心（先去重，再排序）
        centric_list: List[ItemCentric] = list(set(core_item1.get_centric() for core_item1 in core_tuple))
        centric_list.sort(key=lambda x: (x.reduce_name, x.before_handle, x.after_handle))
        centric_tuple = tuple(centric_list)

        # 根据项目集核心进行聚合
        concentric_hash[centric_tuple].append(item1_set)
    return concentric_hash


def merge_same_concentric_item1_set(
        concentric_hash: Dict[Tuple[ItemCentric, ...], List[Item1Set]],
        core_tuple_to_before_item1_set_hash: Dict[Tuple[Item1, ...], List[Tuple[int, Item1Set]]],
        core_tuple_to_item1_set_hash: Dict[Tuple[Item1, ...], Item1Set]
) -> None:
    # pylint: disable=R0914
    """合并同心项目集（原地更新）

    Parameters
    ----------
    concentric_hash : Dict[Tuple[ItemCentric, ...], List[Item1Set]]
        根据项目集核心聚合后的项目集
    core_tuple_to_before_item1_set_hash : Dict[Tuple[Item1, ...], List[Tuple[int, Item1Set]]]
        核心项目元组到该项目集的前置项目集的映射表
    core_tuple_to_item1_set_hash : Dict[Tuple[Item1, ...], Item1Set]
        项目集核心项目元组到项目集闭包的映射
    """
    for _, item1_set_list in concentric_hash.items():
        if len(item1_set_list) == 1:
            continue  # 如果没有项目集核心相同的多个项目集，则不需要合并

        # 构造新的项目集
        new_core_item_set: Set[Item1] = set()  # 新项目集的核心项目
        new_other_item_set: Set[Item1] = set()  # 新项目集的其他等价项目
        for item1_set in item1_set_list:
            for core_item in item1_set.core_tuple:
                new_core_item_set.add(core_item)
            for other_item in item1_set.item_list:
                new_other_item_set.add(other_item)

        # 通过排序逻辑以保证结果状态是稳定的
        new_core_item_list = list(new_core_item_set)
        new_core_item_list.sort()
        new_other_item_list = list(new_other_item_set)
        new_other_item_list.sort()
        new_item1_set = Item1Set.create(
            core_list=tuple(new_core_item_list),
            item_list=new_other_item_list
        )

        # 为新的项目集添加后继项目集；同时更新核心项目元组到该项目集的前置项目集的映射表
        for item1_set in item1_set_list:
            for successor_symbol, successor_item1_set in item1_set.successor_hash.items():
                new_item1_set.set_successor(successor_symbol, successor_item1_set)
                core_tuple_to_before_item1_set_hash[successor_item1_set.core_tuple].remove(
                    (successor_symbol, item1_set))
                core_tuple_to_before_item1_set_hash[successor_item1_set.core_tuple].append(
                    (successor_symbol, new_item1_set))

        # 调整原项目集的前置项目的后继项目集，指向新的项目集；同时更新核心项目元组到该项目集的前置项目集的映射表
        new_before_item_set_list = []  # 新项目集的前置项目集列表
        for item1_set in item1_set_list:
            for successor_symbol, before_item1_set in core_tuple_to_before_item1_set_hash[item1_set.core_tuple]:
                # 此时 before_item1_set 可能已被更新，所以 before_item1_set 的后继项目未必是 item1_set，即不存在：
                # assert before_item1_set.get_successor(successor_symbol).core_tuple == item1_set.core_tuple
                before_item1_set.set_successor(successor_symbol, new_item1_set)
                new_before_item_set_list.append((successor_symbol, before_item1_set))
            core_tuple_to_before_item1_set_hash.pop(item1_set.core_tuple)
        core_tuple_to_before_item1_set_hash[new_item1_set.core_tuple] = new_before_item_set_list

        # 从核心项目到项目集闭包的映射中移除旧项目集，添加新项目集
        for item1_set in item1_set_list:
            core_tuple_to_item1_set_hash.pop(item1_set.core_tuple)
        core_tuple_to_item1_set_hash[new_item1_set.core_tuple] = new_item1_set


class ParserLALR1(ParserBase):
    """LALR(1) 解析器"""

    def __init__(self, grammar: Grammar, debug: bool = False):
        self.symbol_to_start_item_list_hash = None
        super().__init__(grammar, debug=debug)

    def create_action_table_and_goto_table(self):
        # pylint: disable=R0801
        # pylint: disable=R0914
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
        if self.debug is True:
            LOGGER.info("[1 / 10] 计算 Item0 对象开始")
        item0_list = cal_all_item0_list(self.grammar)
        if self.debug is True:
            LOGGER.info(f"[1 / 10] 计算 Item0 对象结束 (Item0 对象数量 = {len(item0_list)})")

        # 根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表
        if self.debug is True:
            LOGGER.info("[2 / 10] 构造非终结符到其初始项目列表的映射表开始")
        symbol_to_start_item_list_hash = cal_symbol_to_start_item_list_hash(item0_list)
        self.symbol_to_start_item_list_hash = symbol_to_start_item_list_hash
        if self.debug is True:
            LOGGER.info(f"[2 / 10] 构造非终结符到其初始项目列表的映射表结束 "
                        f"(映射表元素数量 = {len(self.symbol_to_start_item_list_hash)})")

        # 从项目列表中获取入口项目
        if self.debug is True:
            LOGGER.info("[3 / 10] 从项目列表中获取入口项目开始")
        init_item0 = cal_init_item_from_item_list(item0_list)
        if self.debug is True:
            LOGGER.info("[3 / 10] 从项目列表中获取入口项目结束")

        # 根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射，同时构造项目集闭包之间的关联关系
        if self.debug is True:
            LOGGER.info("[4 / 10] 广度优先搜索，构造项目集闭包之间的关联关系")
        core_tuple_to_item1_set_hash = cal_core_to_item1_set_hash(self.grammar, item0_list, init_item0,
                                                                  symbol_to_start_item_list_hash)
        if self.debug is True:
            LOGGER.info("[4 / 10] 广度优先搜索，构造项目集闭包之间的关联关系结束 "
                        f"(关系映射数量 = {len(core_tuple_to_item1_set_hash)})")

        # 计算核心项目元组到该项目集的前置项目集的映射表
        if self.debug is True:
            LOGGER.info("[5 / 10] 计算核心项目元组到该项目集的前置项目集的映射表开始")
        core_tuple_to_before_item1_set_hash = cal_core_tuple_to_before_item1_set_hash(core_tuple_to_item1_set_hash)
        if self.debug is True:
            LOGGER.info("[5 / 10] 计算核心项目元组到该项目集的前置项目集的映射表结束")

        # 计算项目集核心，并根据项目集的核心（仅包含规约符、符号列表和句柄的核心项目元组）进行聚合
        if self.debug is True:
            LOGGER.info("[6 / 10] 计算项目集核心开始")
        concentric_hash = cal_concentric_hash(core_tuple_to_item1_set_hash)
        if self.debug is True:
            LOGGER.info("[6 / 10] 计算项目集核心结束")

        # 合并项目集核心相同的项目集（原地更新）
        if self.debug is True:
            LOGGER.info("[7 / 10] 合并项目集核心相同的项目集开始")
        merge_same_concentric_item1_set(concentric_hash, core_tuple_to_before_item1_set_hash,
                                        core_tuple_to_item1_set_hash)
        if self.debug is True:
            LOGGER.info("[7 / 10] 合并项目集核心相同的项目集结束")

        # 计算核心项目到项目集闭包 ID（状态）的映射表（增加排序以保证结果状态是稳定的）
        if self.debug is True:
            LOGGER.info("[8 / 10] 计算核心项目到项目集闭包 ID（状态）的映射表开始")
        core_tuple_to_status_hash = {core_tuple: i
                                     for i, core_tuple in enumerate(sorted(core_tuple_to_item1_set_hash, key=repr))}
        if self.debug is True:
            LOGGER.info("[8 / 10] 计算核心项目到项目集闭包 ID（状态）的映射表结束")

        # 生成初始状态
        if self.debug is True:
            LOGGER.info("[9 / 10] 生成初始状态开始")
        init_item1 = Item1.create_by_item0(init_item0, self.grammar.end_terminal)
        entrance_status = core_tuple_to_status_hash[(init_item1,)]
        if self.debug is True:
            LOGGER.info("[9 / 10] 生成初始状态结束")

        # 构造 ACTION 表 + GOTO 表
        if self.debug is True:
            LOGGER.info("[10 / 10] 构造 ACTION 表 + GOTO 表开始")
        accept_item0 = cal_accept_item_from_item_list(item0_list)
        accept_item1 = Item1.create_by_item0(accept_item0, self.grammar.end_terminal)
        accept_item1_set = None
        for core_tuple, item1_set in core_tuple_to_item1_set_hash.items():
            if accept_item1 in core_tuple:
                accept_item1_set = item1_set

        table = create_lr_parsing_table_use_lalr1(
            grammar=self.grammar,
            core_tuple_to_status_hash=core_tuple_to_status_hash,
            core_tuple_to_item1_set_hash=core_tuple_to_item1_set_hash,
            accept_item1_set=accept_item1_set
        )
        if self.debug is True:
            LOGGER.info("[10 / 10] 构造 ACTION 表 + GOTO 表结束")

        return table, entrance_status
