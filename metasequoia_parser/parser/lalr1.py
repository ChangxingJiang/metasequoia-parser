"""
LR(1) 文法解析器
"""

import cProfile
import collections
from typing import Dict, List, Optional, Set, Tuple

from metasequoia_parser.common import Grammar
from metasequoia_parser.common import Item0
from metasequoia_parser.common import Item1
from metasequoia_parser.common import Item1Set
from metasequoia_parser.common import ItemCentric
from metasequoia_parser.common import ItemType
from metasequoia_parser.common import ParserBase
from metasequoia_parser.functions import cal_accept_item_from_item_list
from metasequoia_parser.functions import cal_all_item0_list
from metasequoia_parser.functions import cal_init_item_from_item_list
from metasequoia_parser.functions import cal_symbol_to_start_item_list_hash
from metasequoia_parser.functions import create_lr_parsing_table_use_lalr1
from metasequoia_parser.functions.cal_nonterminal_all_start_terminal import cal_nonterminal_all_start_terminal
from metasequoia_parser.utils import LOGGER
from functools import lru_cache

# 接受（ACCEPT）类型或规约（REDUCE）类型的集合
ACCEPT_OR_REDUCE = {ItemType.ACCEPT, ItemType.REDUCE}


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

    def __init__(self, grammar: Grammar, debug: bool = False, profile_4: Optional[int] = None):
        """

        Parameters
        ----------
        debug : bool, default = False
            【调试】是否开启 Debug 模式日志
        profile_4 : Optional[int], default = None
            【调试】如果不为 None 则开启步骤 4 的 cProfile 性能分析，且广度优先搜索的最大撒次数为 profile_4；如果为 None 则不开启性能分析
        """
        self._profile_4 = profile_4
        self.grammar = grammar
        self.debug = debug

        # 根据文法计算所有项目（Item0 对象），并生成项目之间的后继关系
        if debug is True:
            LOGGER.info("[1 / 10] 计算 Item0 对象开始")
        self.item0_list: List[Item0] = cal_all_item0_list(self.grammar)
        if debug is True:
            LOGGER.info(f"[1 / 10] 计算 Item0 对象结束 (Item0 对象数量 = {len(self.item0_list)})")

        # 根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表
        if self.debug is True:
            LOGGER.info("[2 / 10] 构造非终结符到其初始项目列表的映射表开始")
        self.symbol_to_start_item_list_hash = cal_symbol_to_start_item_list_hash(self.item0_list)
        if self.debug is True:
            LOGGER.info(f"[2 / 10] 构造非终结符到其初始项目列表的映射表结束 "
                        f"(映射表元素数量 = {len(self.symbol_to_start_item_list_hash)})")

        # 从项目列表中获取入口项目
        if self.debug is True:
            LOGGER.info("[3 / 10] 从项目列表中获取入口项目开始")
        self.init_item0 = cal_init_item_from_item_list(self.item0_list)
        if self.debug is True:
            LOGGER.info("[3 / 10] 从项目列表中获取入口项目结束")

        # 计算所有非终结符名称的列表
        nonterminal_name_list = list({item0.nonterminal_id for item0 in self.item0_list})

        # 计算每个非终结符中，所有可能的开头终结符
        self.nonterminal_all_start_terminal = cal_nonterminal_all_start_terminal(self.grammar, nonterminal_name_list)

        # 根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射，同时构造项目集闭包之间的关联关系
        if self.debug is True:
            LOGGER.info("[4 / 10] 广度优先搜索，构造项目集闭包之间的关联关系")
        self.core_tuple_to_item1_set_hash: Dict[Tuple[Item1, ...], Item1Set] = self.cal_core_to_item1_set_hash()
        if self.debug is True:
            LOGGER.info("[4 / 10] 广度优先搜索，构造项目集闭包之间的关联关系结束 "
                        f"(关系映射数量 = {len(self.core_tuple_to_item1_set_hash)})")

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

        # 计算核心项目元组到该项目集的前置项目集的映射表
        if self.debug is True:
            LOGGER.info("[5 / 10] 计算核心项目元组到该项目集的前置项目集的映射表开始")
        core_tuple_to_before_item1_set_hash = cal_core_tuple_to_before_item1_set_hash(self.core_tuple_to_item1_set_hash)
        if self.debug is True:
            LOGGER.info("[5 / 10] 计算核心项目元组到该项目集的前置项目集的映射表结束")

        # 计算项目集核心，并根据项目集的核心（仅包含规约符、符号列表和句柄的核心项目元组）进行聚合
        if self.debug is True:
            LOGGER.info("[6 / 10] 计算项目集核心开始")
        concentric_hash = cal_concentric_hash(self.core_tuple_to_item1_set_hash)
        if self.debug is True:
            LOGGER.info("[6 / 10] 计算项目集核心结束 "
                        f"(项目集核心数量 = {len(concentric_hash)})")

        # 合并项目集核心相同的项目集（原地更新）
        if self.debug is True:
            LOGGER.info("[7 / 10] 合并项目集核心相同的项目集开始")
        merge_same_concentric_item1_set(concentric_hash, core_tuple_to_before_item1_set_hash,
                                        self.core_tuple_to_item1_set_hash)
        if self.debug is True:
            LOGGER.info("[7 / 10] 合并项目集核心相同的项目集结束 "
                        f"(合并后关系映射数量 = {len(self.core_tuple_to_item1_set_hash)})")

        # 计算核心项目到项目集闭包 ID（状态）的映射表（增加排序以保证结果状态是稳定的）
        if self.debug is True:
            LOGGER.info("[8 / 10] 计算核心项目到项目集闭包 ID（状态）的映射表开始")
        core_tuple_to_status_hash = {core_tuple: i
                                     for i, core_tuple in
                                     enumerate(sorted(self.core_tuple_to_item1_set_hash, key=repr))}
        if self.debug is True:
            LOGGER.info("[8 / 10] 计算核心项目到项目集闭包 ID（状态）的映射表结束")

        # 生成初始状态
        if self.debug is True:
            LOGGER.info("[9 / 10] 生成初始状态开始")
        init_item1 = Item1.create_by_item0(self.init_item0, self.grammar.end_terminal)
        entrance_status = core_tuple_to_status_hash[(init_item1,)]
        if self.debug is True:
            LOGGER.info("[9 / 10] 生成初始状态结束")

        # 构造 ACTION 表 + GOTO 表
        if self.debug is True:
            LOGGER.info("[10 / 10] 构造 ACTION 表 + GOTO 表开始")
        accept_item0 = cal_accept_item_from_item_list(self.item0_list)
        accept_item1 = Item1.create_by_item0(accept_item0, self.grammar.end_terminal)
        accept_item1_set = None
        for core_tuple, item1_set in self.core_tuple_to_item1_set_hash.items():
            if accept_item1 in core_tuple:
                accept_item1_set = item1_set

        table = create_lr_parsing_table_use_lalr1(
            grammar=self.grammar,
            core_tuple_to_status_hash=core_tuple_to_status_hash,
            core_tuple_to_item1_set_hash=self.core_tuple_to_item1_set_hash,
            accept_item1_set=accept_item1_set
        )
        if self.debug is True:
            LOGGER.info("[10 / 10] 构造 ACTION 表 + GOTO 表结束")

        return table, entrance_status

    def cal_core_to_item1_set_hash(self) -> Dict[Tuple[Item1, ...], Item1Set]:
        """根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射，同时构造项目集闭包之间的关联关系"""
        # 根据入口项的 LR(0) 项构造 LR(1) 项
        init_item1 = Item1.create_by_item0(self.init_item0, self.grammar.end_terminal)
        init_core_tuple = (init_item1,)

        # 初始化项目集闭包的广度优先搜索的队列：将入口项目集的核心项目元组添加到队列
        visited = {init_core_tuple}
        queue = collections.deque([init_core_tuple])

        # 初始化结果集（项目集核心项目元组到项目集闭包的映射）
        core_tuple_to_item1_set_hash = {}

        # 初始化项目集闭包之间的关联关系（采用个核心项目元组记录）
        item1_set_relation = []

        # 【调试模式】cProfile 性能分析
        profiler = None
        profiler_print = False
        if self._profile_4 is not None:
            profiler = cProfile.Profile()
            profiler.enable()

        # 广度优先搜索遍历所有项目集闭包
        idx = 0
        while queue:
            # 【调试】打印 cProfile 性能分析结果
            if self._profile_4 is not None and profiler_print is False and idx >= self._profile_4:
                profiler.disable()
                profiler.print_stats(sort="time")
                profiler_print = True

            core_tuple = queue.popleft()

            if self.debug is True:
                LOGGER.info(f"正在广度优先搜索遍历所有项目集闭包: 已处理={idx}, 队列中={len(queue)}")

            idx += 1

            # 根据项目集核心项目元组生成项目集闭包中包含的其他项目列表
            item1_list = self.closure_item1(core_tuple)

            # 构造项目集闭包并添加到结果集中
            item1_set = Item1Set.create(core_list=core_tuple, item_list=item1_list)
            core_tuple_to_item1_set_hash[core_tuple] = item1_set

            # 根据后继项目符号进行分组，计算出每个后继项目集闭包的核心项目元组
            successor_group = collections.defaultdict(set)
            for item1 in item1_set.all_item_list:
                if item1.item0.successor_symbol is not None:
                    successor_group[item1.item0.successor_symbol].add(item1.successor_item)

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

        # 【调试】打印 cProfile 性能分析结果
        if self._profile_4 is not None and profiler_print is False:
            profiler.disable()
            profiler.print_stats(sort="time")

        # 构造项目集之间的关系
        for from_core_tuple, successor_symbol, to_core_tuple in item1_set_relation:
            from_item1_set = core_tuple_to_item1_set_hash[from_core_tuple]
            to_item1_set = core_tuple_to_item1_set_hash[to_core_tuple]
            from_item1_set.set_successor(successor_symbol, to_item1_set)
            # print(from_item1_set.core_tuple, "->", to_item1_set.core_tuple)

        return core_tuple_to_item1_set_hash

    def closure_item1(self,
                      core_tuple: Tuple[Item1]
                      ) -> List[Item1]:
        # pylint: disable=R0912
        # pylint: disable=R0914
        """根据项目集核心项目元组（core_tuple）生成项目集闭包中包含的其他项目列表（item_list）

        等价 LR(1) 项目

        Parameters
        ----------
        core_tuple : Tuple[Item1]
            项目集闭包的核心项目（最高层级项目）

        Returns
        -------
        List[Item1]
            项目集闭包中包含的项目列表
        """
        n_terminal = self.grammar.n_terminal  # 【性能】提前获取需频繁使用的 grammar 中的常量，以减少调用次数

        # 初始化项目集闭包中包含的其他项目列表
        item_set: Set[Item1] = set()

        # 初始化广度优先搜索的第 1 批节点
        visited_symbol_set = set()
        queue = collections.deque()
        for item1 in core_tuple:
            if item1.item0.item_type in ACCEPT_OR_REDUCE:
                continue  # 如果核心项是规约项目，则不存在等价项目组，跳过该项目即可

            # 将句柄之后的符号列表 + 展望符添加到队列中
            visited_symbol_set.add((item1.item0.after_handle, item1.lookahead))
            queue.append((item1.item0.after_handle, item1.lookahead))

        # 广度优先搜索所有的等价项目组
        while queue:
            after_handle, lookahead = queue.popleft()

            # 计算单层的等价 LR(1) 项目
            sub_item_set = self.compute_single_level_lr1_closure(
                after_handle=after_handle,
                lookahead=lookahead
            )

            # 将当前项目组匹配的等价项目组添加到所有等价项目组中
            item_set |= sub_item_set

            # 将等价项目组中需要继续寻找等价项目的添加到队列
            for sub_item1 in sub_item_set:
                after_handle = sub_item1.item0.after_handle
                if not after_handle:
                    continue  # 跳过匹配 %empty 的项目

                lookahead = sub_item1.lookahead
                if (after_handle, lookahead) not in visited_symbol_set:
                    visited_symbol_set.add((after_handle, lookahead))
                    queue.append((after_handle, lookahead))

        return list(item_set)

    @lru_cache(maxsize=None)
    def compute_single_level_lr1_closure(self, after_handle: Tuple[int, ...], lookahead: int):
        """计算单层的等价 LR(1) 项目

        Parameters
        ----------
        after_handle : Tuple[int, ...]
            LR(1) 项目在句柄之后的符号的元组
        lookahead : int
            LR(1) 项目的展望符（作为继承的后继符）

        Returns
        -------
        Set[Item1]
            等价 LR(1) 项目的集合
        """
        n_terminal = self.grammar.n_terminal  # 【性能】提前获取需频繁使用的 grammar 中的常量，以减少调用次数

        # 如果开头符号是终结符，则不存在等价项目
        # 【性能】通过 first_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
        first_symbol = after_handle[0]
        if first_symbol < n_terminal:
            return set()

        sub_item_set = set()  # 当前项目组之后的所有可能的 lookahead

        # 添加生成 symbol 非终结符对应的所有项目，并将这些项目也加入继续寻找等价项目组的队列中
        for item0 in self.symbol_to_start_item_list_hash[first_symbol]:
            # 先从当前句柄后第 1 个元素向后继续遍历，添加自生型后继
            i = 1
            is_stop = False  # 是否已经找到不能匹配 %empty 的非终结符或终结符
            while i < len(after_handle):  # 向后逐个遍历符号，寻找展望符
                next_symbol = after_handle[i]

                # 如果遍历到的符号是终结符，则将该终结符添加为展望符，则标记 is_stop 并结束遍历
                # 【性能】通过 next_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
                if next_symbol < n_terminal:
                    sub_item_set.add(Item1.create_by_item0(item0, next_symbol))
                    is_stop = True
                    break

                # 如果遍历到的符号是非终结符，则遍历该非终结符的所有可能的开头终结符添加为展望符
                for start_terminal in self.nonterminal_all_start_terminal[next_symbol]:
                    sub_item_set.add(Item1.create_by_item0(item0, start_terminal))

                # 如果遍历到的非终结符不能匹配 %emtpy，则标记 is_stop 并结束遍历
                if not self.grammar.is_maybe_empty(next_symbol):
                    is_stop = True
                    break

                i += 1

            # 如果没有遍历到不能匹配 %empty 的非终结符或终结符，则添加继承型后继
            if is_stop is False:
                sub_item_set.add(Item1.create_by_item0(item0, lookahead))

        return sub_item_set
