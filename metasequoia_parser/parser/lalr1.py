"""
LR(1) 文法解析器
"""

import cProfile
import collections
import dataclasses
import enum
from functools import lru_cache
from itertools import chain
from typing import Callable, Dict, List, Optional, Set, Tuple

from metasequoia_parser.common import ActionAccept, ActionError, ActionGoto, ActionReduce, ActionShift
from metasequoia_parser.common import CombineType
from metasequoia_parser.common import Grammar
from metasequoia_parser.common import ParserBase
from metasequoia_parser.utils import LOGGER


class ItemType(enum.Enum):
    """项目类型的枚举类"""

    INIT = 0  # 入口项目（Initial Item）：初始文法项目
    ACCEPT = 1  # 接收项目（Accept Item）：解析完成的文法项目

    SHIFT = 2  # 移进项目（Shift Item）：句柄位于开始或中间位置
    REDUCE = 3  # 规约项目（Reduce Item）：句柄位于结束位置，可以规约的项目


@dataclasses.dataclass(slots=True, frozen=True, eq=True, order=True)
class Item0:
    """不提前查看下一个字符的项目类：适用于 LR(0) 解析器和 SLR 解析器

    Attributes
    ----------
    successor_item : Optional[Item0]
        连接到的后继项目对象
    """

    # -------------------- 性能设计 --------------------
    # Item0 项目集唯一 ID
    # 通过在构造时添加 Item0 项目集的唯一 ID，从而将 Item0 项目集的哈希计算优化为直接获取唯一 ID
    i0_id: int = dataclasses.field(kw_only=True, hash=True, compare=False)

    # -------------------- 项目的基本信息（节点属性）--------------------
    nonterminal_id: int = dataclasses.field(kw_only=True, hash=False, compare=True)  # 规约的非终结符 ID（即所在语义组名称对应的 ID）
    before_handle: Tuple[int, ...] = dataclasses.field(kw_only=True, hash=False, compare=True)  # 在句柄之前的符号名称的列表
    ah_id: int = dataclasses.field(kw_only=True, hash=False, compare=True)  # 在句柄之后的符号名称的列表的 ID
    after_handle: Tuple[int, ...] = dataclasses.field(kw_only=True, hash=False,
                                                      compare=False)  # 句柄之后符号名称的列表（用于生成 __repr__）
    item_type: ItemType = dataclasses.field(kw_only=True, hash=False, compare=False)  # 项目类型
    action: Callable = dataclasses.field(kw_only=True, hash=False, compare=False)  # 项目的规约行为函数

    # -------------------- 项目的关联关系（节点出射边）--------------------
    # 能够连接到后继项目的符号名称（即 after_handle 中的第 1 个元素）
    successor_symbol: Optional[int] = dataclasses.field(kw_only=True, hash=False, compare=False)
    successor_item: Optional["Item0"] = dataclasses.field(kw_only=True, hash=False, compare=False)  # 连接到的后继项目对象

    # -------------------- 项目的 SR 优先级、结合方向和 RR 优先级 --------------------
    sr_priority_idx: int = dataclasses.field(kw_only=True, hash=False, compare=False)  # 生成式的 SR 优先级序号（越大越优先）
    sr_combine_type: CombineType = dataclasses.field(kw_only=True, hash=False, compare=False)  # 生成式的 SR 合并顺序
    rr_priority_idx: int = dataclasses.field(kw_only=True, hash=False, compare=False)  # 生成式的 RR 优先级序号（越大越优先）

    def __repr__(self) -> str:
        """将 ItemBase 转换为字符串表示"""
        before_symbol_str = " ".join(str(symbol) for symbol in self.before_handle)
        after_symbol_str = " ".join(str(symbol) for symbol in self.after_handle)
        return f"{self.nonterminal_id}->{before_symbol_str}·{after_symbol_str}"

    def is_init(self) -> bool:
        """是否为入口项目"""
        return self.item_type == ItemType.INIT

    def is_accept(self) -> bool:
        """是否为接收项目"""
        return self.item_type == ItemType.ACCEPT


# 接受（ACCEPT）类型或规约（REDUCE）类型的集合
ACCEPT_OR_REDUCE = {ItemType.ACCEPT, ItemType.REDUCE}


class ParserLALR1(ParserBase):
    """LALR(1) 解析器"""

    def __init__(self, grammar: Grammar, debug: bool = False, profile: bool = False):
        """

        Parameters
        ----------
        debug : bool, default = False
            【调试】是否开启 Debug 模式日志
        profile : Optional[int], default = None
            【调试】如果不为 None 则开启步骤 4 的 cProfile 性能分析，且广度优先搜索的最大撒次数为 profile_4；如果为 None 则不开启性能分析
        """
        self.profile = profile
        self.grammar = grammar
        self.debug = debug

        # 缓存器
        self._dfs_visited = set()

        # 【调试模式】cProfile 性能分析
        self.profiler = None
        if self.profile:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        # LR(0) 项目 ID 到 LR(0) 项目对象的映射
        self.i0_id_to_item0_hash: List[Item0] = []

        # LR(1) 项目核心元组到 LR(1) 项目 ID 的映射
        # - LR(1) 项目核心元组包括指向的 LR(0) 项目 ID 和展望符
        self.item1_core_to_i1_id_hash: Dict[Tuple[int, int], int] = {}

        # 根据文法计算所有项目（Item0 对象），并生成项目之间的后继关系
        LOGGER.info("[1 / 10] 计算 Item0 对象开始")
        self.after_handle_to_ah_id_hash: Dict[Tuple[int, ...], int] = {}  # 句柄之后的符号列表到唯一 ID 的映射
        self.ah_id_to_after_handle_hash: List[Tuple[int, ...]] = []  # 唯一 ID 到句柄之后的符号列表的映射
        self.cal_all_item0_list()
        LOGGER.info(f"[1 / 10] 计算 Item0 对象结束 (Item0 对象数量 = {len(self.i0_id_to_item0_hash)})")

        # 构造每个非终结符到其初始项目（句柄在最左侧）的 LR(0) 项目，即每个备选规则的初始项目的列表的映射表
        # 根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表
        LOGGER.info("[2 / 10] 构造非终结符到其初始项目列表的映射表开始")
        self.symbol_to_start_item_list_hash = self.cal_symbol_to_start_item_list_hash()
        LOGGER.info(f"[2 / 10] 构造非终结符到其初始项目列表的映射表结束 "
                    f"(映射表元素数量 = {len(self.symbol_to_start_item_list_hash)})")

        # 获取入口、接受 LR(0) 项目 ID
        LOGGER.info("[3 / 10] 从 LR(0) 项目列表中获取入口和接受 LR(0) 项目的 ID - 开始")
        self.init_i0_id = self.get_init_i0_id()
        self.accept_i0_id = self.get_accept_i0_id()
        LOGGER.info("[3 / 10] 从 LR(0) 项目列表中获取入口和接受 LR(0) 项目的 ID - 结束")

        # 计算所有非终结符名称的列表
        nonterminal_name_list = list({item0.nonterminal_id for item0 in self.i0_id_to_item0_hash})

        # 计算每个非终结符中，所有可能的开头终结符
        self.nonterminal_all_start_terminal = self.cal_nonterminal_all_start_terminal(nonterminal_name_list)

        # 根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射，同时构造项目集闭包之间的关联关系
        LOGGER.info("[4 / 10] 广度优先搜索，构造项目集闭包之间的关联关系")
        self.item1_set_relation: List[Tuple[int, int, int]] = []  # 核心项目之间的关联关系
        self.core_tuple_to_sid_hash: Dict[Tuple[int, ...], int] = {}  # 核心项目到 SID1 的映射
        self.sid_to_core_tuple_hash: List[Tuple[int, ...]] = []  # SID1 到 LR(1) 项目集核心元组的映射
        self.sid_to_other_i1_id_set_hash: List[Set[int]] = []  # SID1 到 LR(1) 项目集所有元素的映射

        # LR(1) 项目 ID 到指向的 LR(0) 项目 ID 的映射
        self.i1_id_to_i0_id_hash: List[int] = []

        # 查询组合 ID 到句柄之后的符号列表的唯一 ID 与展望符的组合的唯一 ID 的映射
        self.cid_to_ah_id_and_lookahead_list: List[Tuple[int, int]] = []
        self.ah_id_and_lookahead_to_cid_hash: Dict[Tuple[int, int], int] = {}  # 用于构造唯一 ID

        # LR(1) 项目 ID 到组合 ID 的映射
        self.i1_id_to_cid_hash: List[int] = []

        # LR(1) 项目 ID 到展望符的映射
        self.i1_id_to_lookahead_hash: List[int] = []

        # LR(1) 项目 ID 到后继符号及后继 LR(1) 项目 ID
        self.i1_id_to_successor_hash: List[Tuple[int, int]] = []

        # 广度优先搜索，查找 LR(1) 项目集及之间的关联关系
        self.init_i1_id: Optional[int] = None

        # 【临时调试】
        # print("调试结果:",self.cal_all_level_inherited_lr0_by_symbol(161))
        # print("调试结果:", self.cal_single_level_inherited_lr0_by_lr0(518))
        # print("调试结果:", self.cal_single_level_inherited_lr0_by_lr0(519))
        for lr1_id in self.bfs_closure_lr0(113):
            cid = self.i1_id_to_cid_hash[lr1_id]
            ah_id, lookahead = self.cid_to_ah_id_and_lookahead_list[cid]
            after_handle = self.ah_id_to_after_handle_hash[ah_id]
            print(f"[new_closure_lr1] 来源: {after_handle}, {lookahead} (符号 {113} 的自生后继符等价符)")
        # (61, 161, 13), 36
        return

        self.bfs_search_item1_set()

        self.sid_set = set(range(len(self.sid_to_core_tuple_hash)))  # 有效 SID1 的集合
        LOGGER.info("[4 / 10] 广度优先搜索，构造项目集闭包之间的关联关系结束 "
                    f"(搜索后 LR(1) 项目集数量 = {len(self.sid_set)})")

        # 计算项目集核心，并根据项目集的核心（仅包含规约符、符号列表和句柄的核心项目元组）进行聚合
        LOGGER.info("[5 / 10] 计算项目集核心开始")
        self.concentric_hash = self.cal_concentric_hash()
        LOGGER.info(f"[5 / 10] 计算项目集核心结束 (项目集核心数量 = {len(self.concentric_hash)})")

        # 合并项目集核心相同的项目集（原地更新）
        LOGGER.info("[6 / 10] 合并项目集核心相同的项目集开始")
        self.core_tuple_hash = {}
        self.merge_same_concentric_item1_set()
        LOGGER.info("[6 / 10] 合并项目集核心相同的项目集结束 "
                    f"(合并后 LR(1) 项目集数量 = {len(self.sid_set)})")

        # 构造 LR(1) 项目集之间的前驱 / 后继关系
        LOGGER.info("[7 / 10] 构造 LR(1) 项目集之间的前驱 / 后继关系开始")
        self.s1_id_relation = collections.defaultdict(dict)
        self.create_item1_set_relation()
        LOGGER.info("[7 / 10] 构造 LR(1) 项目集之间的前驱 / 后继关系结束")

        # 计算核心项目到项目集闭包 ID（状态）的映射表（增加排序以保证结果状态是稳定的）
        LOGGER.info("[8 / 10] 计算核心项目到项目集闭包 ID（状态）的映射表开始")
        self.sid_to_status_hash = {sid: i for i, sid in enumerate(sorted(self.sid_set))}
        LOGGER.info("[8 / 10] 计算核心项目到项目集闭包 ID（状态）的映射表结束")

        # 计算入口 LR(1) 项目集对应的状态 ID
        LOGGER.info("[9 / 10] 根据入口和接受 LR(1) 项目集对应的状态号")
        accept_i1_id = self.item1_core_to_i1_id_hash[(self.accept_i0_id, self.grammar.end_terminal)]
        self.init_status_id = self.sid_to_status_hash[self.core_tuple_to_sid_hash[(self.init_i1_id,)]]
        self.accept_status_id = self.sid_to_status_hash[self.core_tuple_to_sid_hash[(accept_i1_id,)]]
        LOGGER.info("[9 / 10] 根据入口和接受 LR(1) 项目集对应的状态号")

        # 构造 ACTION 表 + GOTO 表
        LOGGER.info("[10 / 10] 构造 ACTION 表 + GOTO 表开始")
        self.table = self.create_lr_parsing_table_use_lalr1()
        LOGGER.info("[10 / 10] 构造 ACTION 表 + GOTO 表结束")

        if self.profile:
            self.profiler.disable()
            self.profiler.print_stats(sort="cumtime")

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
        return self.table, self.init_status_id

    def cal_all_item0_list(self) -> None:
        """根据文法对象 Grammar 计算出所有项目（Item0 对象）的列表，并生成项目之间的后继关系

        Returns
        -------
        List[Item0]
            所有项目的列表
        """
        # 【性能设计】初始化方法中频繁使用的类属性，以避免重复获取类属性
        grammar: Grammar = self.grammar
        i0_id_to_item0_hash: List[Item0] = self.i0_id_to_item0_hash
        after_handle_to_ah_id_hash: Dict[Tuple[int, ...], int] = self.after_handle_to_ah_id_hash
        ah_id_to_after_handle_hash: List[Tuple[int, ...]] = self.ah_id_to_after_handle_hash

        # 添加空元组的值
        after_handle_to_ah_id_hash[tuple()] = 0
        ah_id_to_after_handle_hash.append(tuple())

        for product in grammar.get_product_list():
            if grammar.is_entrance_symbol(product.nonterminal_id):
                # 当前生成式是入口生成式
                last_item_type = ItemType.ACCEPT
                first_item_type = ItemType.INIT
            else:
                last_item_type = ItemType.REDUCE
                first_item_type = ItemType.SHIFT

            # 如果为 %empty，则仅构造一个规约项目
            if len(product.symbol_id_list) == 0:
                i0_id = len(i0_id_to_item0_hash)
                i0_id_to_item0_hash.append(Item0(
                    i0_id=i0_id,
                    nonterminal_id=product.nonterminal_id,
                    before_handle=tuple(),
                    ah_id=0,
                    after_handle=tuple(),
                    action=product.action,
                    item_type=last_item_type,
                    successor_symbol=None,  # 规约项目不存在后继项目
                    successor_item=None,  # 规约项目不存在后继项目
                    sr_priority_idx=product.sr_priority_idx,
                    sr_combine_type=product.sr_combine_type,
                    rr_priority_idx=product.rr_priority_idx
                ))
                continue

            # 添加句柄在结束位置（最右侧）的项目（规约项目）
            i0_id = len(i0_id_to_item0_hash)
            last_item = Item0(
                i0_id=i0_id,
                nonterminal_id=product.nonterminal_id,
                before_handle=tuple(product.symbol_id_list),
                ah_id=0,
                after_handle=tuple(),
                action=product.action,
                item_type=last_item_type,
                successor_symbol=None,  # 规约项目不存在后继项目
                successor_item=None,  # 规约项目不存在后继项目
                sr_priority_idx=product.sr_priority_idx,
                sr_combine_type=product.sr_combine_type,
                rr_priority_idx=product.rr_priority_idx
            )
            i0_id_to_item0_hash.append(last_item)

            # 从右向左依次添加句柄在中间位置（不是最左侧和最右侧）的项目（移进项目），并将上一个项目作为下一个项目的后继项目
            for i in range(len(product.symbol_id_list) - 1, 0, -1):
                i0_id = len(i0_id_to_item0_hash)
                after_handle = tuple(product.symbol_id_list[i:])
                if after_handle not in after_handle_to_ah_id_hash:
                    ah_id = len(ah_id_to_after_handle_hash)
                    after_handle_to_ah_id_hash[after_handle] = ah_id
                    ah_id_to_after_handle_hash.append(after_handle)
                else:
                    ah_id = after_handle_to_ah_id_hash[after_handle]
                now_item = Item0(
                    i0_id=i0_id,
                    nonterminal_id=product.nonterminal_id,
                    before_handle=tuple(product.symbol_id_list[:i]),
                    ah_id=ah_id,
                    after_handle=after_handle,
                    action=product.action,
                    item_type=ItemType.SHIFT,
                    successor_symbol=product.symbol_id_list[i],
                    successor_item=last_item,
                    sr_priority_idx=product.sr_priority_idx,
                    sr_combine_type=product.sr_combine_type,
                    rr_priority_idx=product.rr_priority_idx
                )
                i0_id_to_item0_hash.append(now_item)
                last_item = now_item

            # 添加添加句柄在开始位置（最左侧）的项目（移进项目或入口项目）
            i0_id = len(i0_id_to_item0_hash)
            after_handle = tuple(product.symbol_id_list)
            if after_handle not in after_handle_to_ah_id_hash:
                ah_id = len(ah_id_to_after_handle_hash)
                after_handle_to_ah_id_hash[after_handle] = ah_id
                ah_id_to_after_handle_hash.append(after_handle)
            else:
                ah_id = after_handle_to_ah_id_hash[after_handle]
            i0_id_to_item0_hash.append(Item0(
                i0_id=i0_id,
                nonterminal_id=product.nonterminal_id,
                before_handle=tuple(),
                ah_id=ah_id,
                after_handle=after_handle,
                action=product.action,
                item_type=first_item_type,
                successor_symbol=product.symbol_id_list[0],
                successor_item=last_item,
                sr_priority_idx=product.sr_priority_idx,
                sr_combine_type=product.sr_combine_type,
                rr_priority_idx=product.rr_priority_idx
            ))

    def cal_symbol_to_start_item_list_hash(self) -> Dict[int, List[Item0]]:
        """根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表
        Returns
        -------
        Dict[int, List[T]]
            键为非终结符名称，值为非终结符对应项目的列表（泛型 T 为 ItemBase 的子类）
        """
        symbol_to_start_item_list_hash: Dict[int, List[Item0]] = collections.defaultdict(list)
        for item0 in self.i0_id_to_item0_hash:
            if not item0.before_handle:
                symbol_to_start_item_list_hash[item0.nonterminal_id].append(item0)
        return symbol_to_start_item_list_hash

    def get_init_i0_id(self) -> int:
        """从 LR(0) 项目列表中获取入口 LR(0) 项目的 ID"""
        for item0 in self.i0_id_to_item0_hash:
            if item0.is_init():
                return item0.i0_id
        raise KeyError("未从项目列表中获取到 INIT 项目")

    def get_accept_i0_id(self) -> int:
        """从 LR(0) 项目列表中获取入口 LR(0) 项目的 ID"""
        for item0 in self.i0_id_to_item0_hash:
            if item0.is_accept():
                return item0.i0_id
        raise KeyError("未从项目列表中获取到 ACCEPT 项目")

    def _create_item1(self, item0: Item0, lookahead: int) -> int:
        """如果 item1 不存在则构造 item1，返回直接返回已构造的 item1"""

        # 如果 item1 已经存在则返回已存在 item1
        if (item0.i0_id, lookahead) in self.item1_core_to_i1_id_hash:
            return self.item1_core_to_i1_id_hash[(item0.i0_id, lookahead)]

        if item0.successor_item is not None:
            successor_item1 = self._create_item1(item0.successor_item, lookahead)
        else:
            successor_item1 = None

        i1_id = len(self.item1_core_to_i1_id_hash)
        self.item1_core_to_i1_id_hash[(item0.i0_id, lookahead)] = i1_id
        self.i1_id_to_i0_id_hash.append(item0.i0_id)
        self.i1_id_to_lookahead_hash.append(lookahead)

        if (item0.ah_id, lookahead) not in self.ah_id_and_lookahead_to_cid_hash:
            cid = len(self.ah_id_and_lookahead_to_cid_hash)
            self.ah_id_and_lookahead_to_cid_hash[(item0.ah_id, lookahead)] = cid
            self.cid_to_ah_id_and_lookahead_list.append((item0.ah_id, lookahead))
        else:
            cid = self.ah_id_and_lookahead_to_cid_hash[(item0.ah_id, lookahead)]
        self.i1_id_to_cid_hash.append(cid)

        self.i1_id_to_successor_hash.append((item0.successor_symbol, successor_item1))
        return i1_id

    def cal_nonterminal_all_start_terminal(self, nonterminal_name_list: List[int]) -> Dict[int, Set[int]]:
        """计算每个非终结符中，所有可能的开头终结符

        Parameters
        ----------
        nonterminal_name_list : List[int]
            所有非终结符名称的列表

        Returns
        -------
        Dict[int, Set[int]]
            每个非终结标识符到其所有可能的开头终结符集合的映射
        """
        # 【性能设计】初始化方法中频繁使用的类属性，以避免重复获取类属性
        grammar: Grammar = self.grammar

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

    def bfs_search_item1_set(self) -> None:
        """根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射，同时构造项目集闭包之间的关联关系"""

        # 【性能设计】初始化方法中频繁使用的类属性，以避免重复获取类属性
        core_tuple_to_sid_hash = self.core_tuple_to_sid_hash
        sid_to_core_tuple_hash = self.sid_to_core_tuple_hash
        item1_set_relation = self.item1_set_relation
        i1_id_to_successor_hash = self.i1_id_to_successor_hash

        # 根据入口项的 LR(0) 项构造 LR(1) 项
        self.init_i1_id = self._create_item1(self.i0_id_to_item0_hash[self.init_i0_id], self.grammar.end_terminal)
        init_core_tuple = (self.init_i1_id,)
        core_tuple_to_sid_hash[init_core_tuple] = 0
        sid_to_core_tuple_hash.append(init_core_tuple)

        # 初始化项目集闭包的广度优先搜索的队列：将入口项目集的核心项目元组添加到队列
        visited = {0}
        queue = collections.deque([0])

        # 广度优先搜索遍历所有项目集闭包
        idx = 0
        while queue:
            sid1 = queue.popleft()
            core_tuple = sid_to_core_tuple_hash[sid1]

            if self.debug is True:
                LOGGER.info(f"正在广度优先搜索遍历所有项目集闭包: 已处理={idx}, 队列中={len(queue)}")

            idx += 1

            # 广度优先搜索，根据项目集核心项目元组（core_tuple）生成项目集闭包中包含的其他项目列表（item_list）
            if core_tuple == (19762, 19769, 19778):
                debug = True
            else:
                debug = False
            other_i1_id_set = self.new_closure_lr1(core_tuple, debug)
            # other_i1_id_set = self.bfs_closure_item1(core_tuple)
            if other_i1_id_set != self.bfs_closure_item1(core_tuple):
                if debug:
                    print(f"【DIFF】core_tuple={core_tuple}")
                    print(list(sorted(other_i1_id_set)))
                    print(list(sorted(self.bfs_closure_item1(core_tuple))))
                    for i1_id in core_tuple:
                        cid = self.i1_id_to_cid_hash[i1_id]
                        ah_id, lookahead = self.cid_to_ah_id_and_lookahead_list[cid]
                        after_handle = self.ah_id_to_after_handle_hash[ah_id]
                        print(f"【DIFF】参数：{after_handle}, {lookahead}")

                    for i1_id in self.bfs_closure_item1(core_tuple) - other_i1_id_set:
                        cid = self.i1_id_to_cid_hash[i1_id]
                        ah_id, lookahead = self.cid_to_ah_id_and_lookahead_list[cid]
                        after_handle = self.ah_id_to_after_handle_hash[ah_id]
                        print(f"【DIFF】（缺少）{after_handle}, {lookahead}")

                    for i1_id in other_i1_id_set - self.bfs_closure_item1(core_tuple):
                        cid = self.i1_id_to_cid_hash[i1_id]
                        ah_id, lookahead = self.cid_to_ah_id_and_lookahead_list[cid]
                        after_handle = self.ah_id_to_after_handle_hash[ah_id]
                        print(f"【DIFF】（多出）{after_handle}, {lookahead}")

            # 构造项目集闭包并添加到结果集中
            self.sid_to_other_i1_id_set_hash.append(other_i1_id_set)

            # 根据后继项目符号进行分组，计算出每个后继项目集闭包的核心项目元组
            successor_group = collections.defaultdict(list)
            for i1_id in chain(core_tuple, other_i1_id_set):
                successor_symbol, successor_i1_id = i1_id_to_successor_hash[i1_id]
                if successor_symbol is not None:
                    successor_group[successor_symbol].append(successor_i1_id)

            # 计算后继项目集的核心项目元组（排序以保证顺序稳定）
            for successor_symbol, sub_i1_id_set in successor_group.items():
                successor_core_tuple: Tuple[int, ...] = tuple(sorted(set(sub_i1_id_set)))
                if successor_core_tuple not in core_tuple_to_sid_hash:
                    successor_sid1 = len(core_tuple_to_sid_hash)
                    core_tuple_to_sid_hash[successor_core_tuple] = successor_sid1
                    sid_to_core_tuple_hash.append(successor_core_tuple)
                else:
                    successor_sid1 = core_tuple_to_sid_hash[successor_core_tuple]

                # 记录 LR(1) 项目集之间的前驱 / 后继关系
                item1_set_relation.append((sid1, successor_symbol, successor_sid1))

                # 将后继项目集闭包的核心项目元组添加到队列
                if successor_sid1 not in visited:
                    queue.append(successor_sid1)
                    visited.add(successor_sid1)

    def new_closure_lr1(self, core_tuple: Tuple[int], debug: bool = False) -> Set[int]:
        """新版项目集闭包计算方法

        【样例 1】S->A·B,c（其中 B 不可为空）
        1. 处理 B：添加 B 所有自生后继符等价项

        【样例 2】S->A·B,c（其中 B 可为空）
        1. 处理 B：添加 B 所有自生后继等价项，将 B 的所有继承后继等价项添加到等待集合中
        2. 处理 c：将等待集合中的所有元素与 c 构造继承后继等价项

        【样例 3】S->A·Bc,d（其中 B 可为空）
        1. 处理 B: 添加 B 的所有自生后继符等价项，将 B 的所有继承后继符等价项添加到等待集合中
        2. 处理 c：将等待集合中的所有元素与 c 构造继承后继等价项

        【样例 4】S->A·Bc,d（其中 B 不可为空）
        1. 处理 B: 添加 B 的所有自生后继符等价项，将 B 的所有继承后继符等价项添加到等待集合中
        2. 处理 c：将等待集合中的所有元素与 c 构造继承后继等价项

        【样例 5】S->A·BC,d（其中 B 可为空、C 不可为空）
        1. 处理 B: 添加 B 的所有自生后继符等价项，将 B 的所有继承后继符等价项添加到等待后继符项集合中
        2. 处理 C：添加 C 的所有自生后继符等价项，将等待集合中的所有元素与 C 的所有开头终结符构造继承后继等价项

        【样例 6】S->A·BC,d（其中 B 可为空、C 可为空）
        1. 处理 B: 添加 B 的所有自生后继符等价项，将 B 的所有继承后继符等价项添加到等待后继符项集合中
        2. 处理 C：添加 C 的所有自生后继符等价项，将等待集合中的所有元素与 C 的所有开头终结符构造继承后继等价项，将 C 的所有继承后继等价项添加到等待集合中
        3. 处理 d：将等待集合中的所有元素与 d 构造继承后继等价项

        【处理逻辑】
        逐个遍历 after_handle 中的符号和展望符，对每个元素执行如下逻辑：
        - 如果当前符号是终结符：
          - 将等待集合中的元素与该终结符构造继承后继等价项
          - 结束项目集闭包计算
        - 如果当前符号是非终结符
          - 添加当前非终结符的所有自生后继符等价项
          - 将等待集合中的所有元素与当前非终结符的所有可能开头终结符构造继承后继等价项
          - 如果当前非终结符可匹配空（%empty）：
            - 将当前非终结符的所有继承后继符等价项添加到等待集合
          - 如果当前非终结符不可匹配空（%empty）：
            - 结束项目集闭包计算

        【实现策略】
        缓存非终结符的所有自生后继符等价项，即非终结符 ID 到 LR(1) 的集合的映射
        缓存非终结符的所有继承后继符等价项，即非终结符 ID 到等待展望符的 LR(0) 的集合的映射
        提前计算非终结符的所有可能的开头终结符
        """
        lr1_id_set = set()
        for lr1_id in core_tuple:
            cid = self.i1_id_to_cid_hash[lr1_id]
            ah_id, lookahead = self.cid_to_ah_id_and_lookahead_list[cid]
            after_handle = self.ah_id_to_after_handle_hash[ah_id]

            # 合并 after_handle 和展望符，此时最后一个字符一定是终结符
            assert self.grammar.is_terminal(lookahead), (f"展望符不是终结符: lookahead={lookahead}, "
                                                         f"after_hande={after_handle}")
            merge_after_handle = list(after_handle) + [lookahead]

            if debug:
                print(f"[new_closure_lr1] ------------------------------------------------------")
                print(f"[new_closure_lr1] after_handle={after_handle}, lookahead={lookahead}")

            waiting_lr0_id_set = set()
            for i, symbol in enumerate(merge_after_handle):
                if debug:
                    print(f"[new_closure_lr1] "
                          f"symbol: {symbol}, "
                          f"is_terminal={self.grammar.is_terminal(symbol)}, "
                          f"is_maybe_empty={self.grammar.is_maybe_empty(symbol)}")
                # 如果当前符号是终结符，则将等待集合中的元素与该终结符构造继承后继等价项，然后结束符号项目集闭包
                if self.grammar.is_terminal(symbol):
                    for lr0_id in waiting_lr0_id_set:
                        item0 = self.i0_id_to_item0_hash[lr0_id]
                        if debug:
                            print(f"[new_closure_lr1] 来源: {item0.after_handle}, {symbol} (等待集合与当前终结符 {symbol})")
                        lr1_id_set.add(self._create_item1(self.i0_id_to_item0_hash[lr0_id], symbol))
                    break

                # 如果当前符号是非终结符
                # 添加当前非终结符的所有自生后继符等价项
                lr1_id_set |= self.bfs_closure_lr0(symbol)
                if debug:
                    for lr1_id in self.bfs_closure_lr0(symbol):
                        cid = self.i1_id_to_cid_hash[lr1_id]
                        ah_id, lookahead = self.cid_to_ah_id_and_lookahead_list[cid]
                        after_handle = self.ah_id_to_after_handle_hash[ah_id]
                        print(f"[new_closure_lr1] 来源: {after_handle}, {lookahead} (符号 {symbol} 的自生后继符等价符)")

                # 将等待集合中的所有元素与当前非终结符的所有可能开头终结符构造继承后继等价项
                for lr0_id in waiting_lr0_id_set:
                    for start_terminal in self.nonterminal_all_start_terminal[symbol]:
                        if debug:
                            item0 = self.i0_id_to_item0_hash[lr0_id]
                            print(f"[new_closure_lr1] 来源: {item0.after_handle}, {start_terminal} (等待集合与当前终结符 {symbol} 的开头终结符)")

                        lr1_id_set.add(self._create_item1(self.i0_id_to_item0_hash[lr0_id], start_terminal))

                # 如果当前非终结符可匹配空（%empty）：
                # 将当前非终结符的所有继承后继符等价项添加到等待集合
                waiting_lr0_id_set |= self.cal_all_level_inherited_lr0_by_symbol(symbol)
                if debug:
                    print(f"[new_closure_lr1]"
                          f" waiting_lr0_id_set: {[self.i0_id_to_item0_hash[lr0_id] for lr0_id in waiting_lr0_id_set]}")
                    # print(f"[new_closure_lr1]"
                    #       f" lr1_id_set: {[self.ah_id_to_after_handle_hash[[0]] for lr1_id in lr1_id_set]}")
                #     new_lr0_id_set = self.cal_all_level_inherited_lr0_by_symbol(symbol)
                #     for lr0_id in new_lr0_id_set:
                #         item0 = self.i0_id_to_item0_hash[lr0_id]
                #         print(lr0_id, item0.nonterminal_id, ">", item0.before_handle, "·", item0.after_handle)
                # 如果当前非终结符不可匹配空（%empty）：
                if not self.grammar.is_maybe_empty(symbol):
                    assert i + 1 < len(merge_after_handle)
                    for sub_symbol in merge_after_handle[i + 1:]:
                        if self.grammar.is_terminal(sub_symbol):
                            for lr0_id in waiting_lr0_id_set:
                                if debug:
                                    item0 = self.i0_id_to_item0_hash[lr0_id]
                                    print(
                                        f"[new_closure_lr1] 来源: {item0.after_handle}, {sub_symbol} (之后的终结符)")
                                lr1_id_set.add(self._create_item1(self.i0_id_to_item0_hash[lr0_id], sub_symbol))
                            break
                        else:
                            for lr0_id in waiting_lr0_id_set:
                                for start_terminal in self.nonterminal_all_start_terminal[sub_symbol]:
                                    if debug:
                                        item0 = self.i0_id_to_item0_hash[lr0_id]
                                        print(
                                            f"[new_closure_lr1] 来源: {item0.after_handle}, {start_terminal} (之后的非终结符 {sub_symbol} 开头的终结符)")
                                    lr1_id_set.add(self._create_item1(self.i0_id_to_item0_hash[lr0_id], start_terminal))
                            if not self.grammar.is_maybe_empty(sub_symbol):
                                break
                            # next_symbol = merge_after_handle[i + 1]
                            # for lr0_id in waiting_lr0_id_set:
                            #     lr1_id_set.add(self._create_item1(self.i0_id_to_item0_hash[lr0_id], next_symbol))
                    break

        return lr1_id_set

    @lru_cache(maxsize=None)
    def cal_all_level_inherited_lr0_by_symbol(self, symbol: int) -> Set[int]:
        """根据非终结符，计算所有层级的继承后继型等价 LR(1) 项目对应 LR(0) 项目 ID 的集合"""
        lr0_id_set = set()
        for lr0 in self.symbol_to_start_item_list_hash[symbol]:
            lr0_id_set |= self.cal_all_level_inherited_lr0_by_lr0(lr0.i0_id)
        return lr0_id_set

    @lru_cache(maxsize=None)
    def cal_all_level_inherited_lr0_by_lr0(self, lr0_id: int) -> Set[int]:
        """根据 LR(0) 项目，计算所有层级的继承后继型等价 LR(1) 项目对应 LR(0) 项目 ID 的集合【包含自身】"""
        visited = {lr0_id}
        queue = collections.deque([lr0_id])
        while queue:
            # 根据 LR(0) 项目，计算单一层级的继承后继型等价 LR(1) 项目对应 LR(0) 项目的集合
            sub_lr0_list = self.cal_single_level_inherited_lr0_by_lr0(queue.popleft())
            # print(f"[cal_all_level_inherited_lr0_by_lr0] {queue}: {sub_lr0_list}")

            for lr0 in sub_lr0_list:
                if lr0.i0_id not in visited:
                    visited.add(lr0.i0_id)
                    queue.append(lr0.i0_id)

        return visited

    @lru_cache(maxsize=None)
    def cal_single_level_inherited_lr0_by_lr0(self, lr0_id: int) -> List[Item0]:
        """根据 LR(0) 项目，计算单一层级的继承后继型等价 LR(1) 项目对应 LR(0) 项目的集合【可能包含自身】"""
        lr0 = self.i0_id_to_item0_hash[lr0_id]
        after_handle = lr0.after_handle

        # print(f"[cal_single_level_inherited_lr0_by_lr0] {lr0_id}: {lr0}")

        if not after_handle:
            return []

        # 获取当前句柄之后的第 1 个符号
        first_symbol = after_handle[0]

        # 如果当前句柄之后的第 1 个符号是终结符，则不存在等价的 LR(1) 项目
        if first_symbol < self.grammar.n_terminal:
            return []

        # 如果句柄之后的任何符号是终结符或不允许为空，则不存在等价的继承后继型等价 LR(1) 项目
        after_handle = lr0.after_handle
        for sub_symbol in after_handle[1:]:
            if self.grammar.is_terminal(sub_symbol):
                return []
            if not self.grammar.is_maybe_empty(sub_symbol):
                return []
        return self.symbol_to_start_item_list_hash[first_symbol]

    def bfs_closure_item1(self, core_tuple: Tuple[int]) -> Set[int]:
        # pylint: disable=R0912
        # pylint: disable=R0914
        """广度优先搜索，根据项目集核心项目元组（core_tuple）生成项目集闭包中包含的其他项目列表（item_list）

        返回 LR(1) 项目的 ID 的集合

        【性能设计】这里采用广度优先搜索，是因为当 core_tuple 中包含多个 LR(1) 项目时，各个 LR(1) 项目的等价 LR(1) 项目之间往往会存在大量相同
        的元素；如果采用深度优先搜索，那么在查询缓存、合并结果、检查搜索条件是否相同时，会进行非常多的 Tuple[Item1, ...] 比较，此时会消耗更多的性
        能。然而，不同 core_tuple 之间，相同的 LR(1) 项目的数量可能反而较少。因此，虽然广度优先搜索在时间复杂度上劣于深度优先搜索，但是经过测试在
        当前场景下的性能是优于深度优先搜索的。

        Parameters
        ----------
        core_tuple : Tuple[int]
            项目集闭包的核心项目（最高层级项目）

        Returns
        -------
        List[int]
            项目集闭包中包含的项目列表
        """
        i1_id_to_cid_hash = self.i1_id_to_cid_hash  # 【性能设计】将实例变量缓存为局部遍历那个

        # 初始化项目集闭包中包含的其他项目列表
        i1_id_set: Set[int] = set()

        # 初始化广度优先搜索的第 1 批节点
        visited_cid_set = {i1_id_to_cid_hash[i1_id] for i1_id in core_tuple}
        queue = collections.deque(visited_cid_set)

        # 广度优先搜索所有的等价项目组
        while queue:
            # 计算单层的等价 LR(1) 项目
            sub_i1_id_set = self.compute_single_level_lr1_closure(queue.popleft())

            # 将当前项目组匹配的等价项目组添加到所有等价项目组中
            i1_id_set |= sub_i1_id_set

            # 将等价项目组中需要继续寻找等价项目的添加到队列
            # 【性能设计】在这里没有使用更 Pythonic 的批量操作，是因为批量操作会至少创建 2 个额外的集合，且会额外执行一次哈希计算，这带来的外性能消耗超过了 Python 循环和判断的额外消耗
            for i1_id in sub_i1_id_set:
                new_cid = i1_id_to_cid_hash[i1_id]
                if new_cid not in visited_cid_set:
                    visited_cid_set.add(new_cid)
                    queue.append(new_cid)

        return i1_id_set

    @lru_cache(maxsize=None)
    def compute_single_level_lr1_closure(self, cid: int) -> Set[int]:
        """计算 item1 单层的等价 LR(1) 项目的 ID 的集合

        计算单层的等价 LR(1) 项目，即只将非终结符替换为等价的终结符或非终结符，但不会计算替换后的终结符的等价 LR(1) 项目。

        Parameters
        ----------
        cid : int
            组合 ID

        Returns
        -------
        Set[int]
            等价 LR(1) 项目的集合
        """
        ah_id, lookahead = self.cid_to_ah_id_and_lookahead_list[cid]

        # 如果是规约项目，则一定不存在等价项目组，跳过该项目即可
        if ah_id == 0:
            return set()

        n_terminal = self.grammar.n_terminal  # 【性能设计】提前获取需频繁使用的 grammar 中的常量，以减少调用次数
        after_handle = self.ah_id_to_after_handle_hash[ah_id]
        len_after_handle = len(after_handle)  # 【性能设计】提前计算需要频繁使用的常量

        # 获取当前句柄之后的第 1 个符号
        first_symbol = after_handle[0]

        # 如果当前句柄之后的第 1 个符号是终结符，则不存在等价的 LR(1) 项目，直接返回空集合
        # 【性能】通过 first_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
        if first_symbol < n_terminal:
            return set()

        sub_item_set: Set[int] = set()  # 当前项目组之后的所有可能的 lookahead

        # 添加生成 symbol 非终结符对应的所有项目，并将这些项目也加入继续寻找等价项目组的队列中
        # 先从当前句柄后第 1 个元素向后继续遍历，添加自生型后继
        i = 1
        is_stop = False  # 是否已经找到不能匹配 %empty 的非终结符或终结符
        while i < len_after_handle:  # 向后逐个遍历符号，寻找展望符
            next_symbol = after_handle[i]

            # 如果遍历到的符号是终结符，则将该终结符添加为展望符，则标记 is_stop 并结束遍历
            # 【性能】通过 next_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
            if next_symbol < n_terminal:
                sub_item_set |= self.create_lr1_by_symbol_and_lookahead(first_symbol, next_symbol)  # 自生后继符
                is_stop = True
                break

            # 如果遍历到的符号是非终结符，则遍历该非终结符的所有可能的开头终结符添加为展望符
            for start_terminal in self.nonterminal_all_start_terminal[next_symbol]:
                sub_item_set |= self.create_lr1_by_symbol_and_lookahead(first_symbol, start_terminal)  # 自生后继符

            # 如果遍历到的非终结符不能匹配 %emtpy，则标记 is_stop 并结束遍历
            if not self.grammar.is_maybe_empty(next_symbol):
                is_stop = True
                break

            i += 1

        # 如果没有遍历到不能匹配 %empty 的非终结符或终结符，则添加继承型后继
        if is_stop is False:
            sub_item_set |= self.create_lr1_by_symbol_and_lookahead(first_symbol, lookahead)  # 继承后继符

        return sub_item_set

    @lru_cache(maxsize=None)
    def bfs_closure_lr0(self, symbol: int) -> Set[int]:
        # pylint: disable=R0912
        # pylint: disable=R0914
        """根据 LR(0) 项目，计算其对应的所有自生后继型 LR(1) 项目

        Returns
        -------
        List[int]
            项目集闭包中包含的项目列表
        """
        # print(f"[bfs_closure_lr0] symbol={symbol}")
        # 初始化广度优先搜索的第 1 批节点
        visited_cid_set = set()
        queue = collections.deque()
        for lr0 in self.symbol_to_start_item_list_hash[symbol]:
            ah_id = self.after_handle_to_ah_id_hash[lr0.after_handle]
            queue.append((ah_id, None))
            visited_cid_set.add((ah_id, None))

        # 广度优先搜索所有的等价项目组
        i1_id_set = set()
        while queue:
            ah_id, lookahead = queue.popleft()

            # 计算单层的等价 LR(1) 项目
            sub_i1_id_set = self.compute_single_level_lr1_closure_2(ah_id, lookahead)
            # if symbol == 113:
            #     after_handle = self.ah_id_to_after_handle_hash[ah_id]
            #     print(f"[bfs_closure_lr0] compute_single_level_lr1_closure_2: {ah_id}{after_handle}, {lookahead} -> {sub_i1_id_set}")

            # 将当前项目组匹配的等价项目组添加到所有等价项目组中
            for sub_i1_id in sub_i1_id_set:
                sub_lookahead = self.i1_id_to_lookahead_hash[sub_i1_id]
                if symbol == 113:
                    after_handle = self.ah_id_to_after_handle_hash[ah_id]
                    new_cid = self.i1_id_to_cid_hash[sub_i1_id]
                    sub_ah_id, _ = self.cid_to_ah_id_and_lookahead_list[new_cid]
                    sub_after_handle = self.ah_id_to_after_handle_hash[sub_ah_id]
                    print(f"[bfs_closure_lr0] {after_handle}, {lookahead} > {sub_after_handle}, {sub_lookahead}")
                if sub_lookahead is not None:
                    i1_id_set.add(sub_i1_id)

            # 将等价项目组中需要继续寻找等价项目的添加到队列
            # 【性能设计】在这里没有使用更 Pythonic 的批量操作，是因为批量操作会至少创建 2 个额外的集合，且会额外执行一次哈希计算，这带来的外性能消耗超过了 Python 循环和判断的额外消耗
            for i1_id in sub_i1_id_set:
                new_cid = self.i1_id_to_cid_hash[i1_id]
                ah_id, lookahead = self.cid_to_ah_id_and_lookahead_list[new_cid]
                if (ah_id, lookahead) not in visited_cid_set:
                    visited_cid_set.add((ah_id, lookahead))
                    queue.append((ah_id, lookahead))
        # print(f"bfs_closure_lr0: {symbol} -> {i1_id_set}")
        return i1_id_set

    @lru_cache(maxsize=None)
    def compute_single_level_lr1_closure_2(self, ah_id: int, lookahead: Optional[int]) -> Set[int]:
        """计算 item1 单层的等价 LR(1) 项目的 ID 的集合

        计算单层的等价 LR(1) 项目，即只将非终结符替换为等价的终结符或非终结符，但不会计算替换后的终结符的等价 LR(1) 项目。

        Parameters
        ----------
        cid : int
            组合 ID

        Returns
        -------
        Set[int]
            等价 LR(1) 项目的集合
        """

        # 如果是规约项目，则一定不存在等价项目组，跳过该项目即可
        if ah_id == 0:
            return set()

        n_terminal = self.grammar.n_terminal  # 【性能设计】提前获取需频繁使用的 grammar 中的常量，以减少调用次数
        after_handle = self.ah_id_to_after_handle_hash[ah_id]
        len_after_handle = len(after_handle)  # 【性能设计】提前计算需要频繁使用的常量

        # 获取当前句柄之后的第 1 个符号
        first_symbol = after_handle[0]

        # 如果当前句柄之后的第 1 个符号是终结符，则不存在等价的 LR(1) 项目，直接返回空集合
        # 【性能】通过 first_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
        if first_symbol < n_terminal:
            return set()

        sub_item_set: Set[int] = set()  # 当前项目组之后的所有可能的 lookahead

        # 添加生成 symbol 非终结符对应的所有项目，并将这些项目也加入继续寻找等价项目组的队列中
        # 先从当前句柄后第 1 个元素向后继续遍历，添加自生型后继
        i = 1
        is_stop = False  # 是否已经找到不能匹配 %empty 的非终结符或终结符
        while i < len_after_handle:  # 向后逐个遍历符号，寻找展望符
            next_symbol = after_handle[i]

            # 如果遍历到的符号是终结符，则将该终结符添加为展望符，则标记 is_stop 并结束遍历
            # 【性能】通过 next_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
            if next_symbol < n_terminal:
                sub_item_set |= self.create_lr1_by_symbol_and_lookahead(first_symbol, next_symbol)  # 自生后继符
                is_stop = True
                break

            # 如果遍历到的符号是非终结符，则遍历该非终结符的所有可能的开头终结符添加为展望符
            for start_terminal in self.nonterminal_all_start_terminal[next_symbol]:
                sub_item_set |= self.create_lr1_by_symbol_and_lookahead(first_symbol, start_terminal)  # 自生后继符

            # 如果遍历到的非终结符不能匹配 %emtpy，则标记 is_stop 并结束遍历
            if not self.grammar.is_maybe_empty(next_symbol):
                is_stop = True
                break

            i += 1

        # 如果没有遍历到不能匹配 %empty 的非终结符或终结符，则添加继承型后继
        if is_stop is False:
            sub_item_set |= self.create_lr1_by_symbol_and_lookahead(first_symbol, lookahead)  # 继承后继符

        return sub_item_set

    @lru_cache(maxsize=None)
    def create_lr1_by_symbol_and_lookahead(self, symbol: int, lookahead: int) -> Set[int]:
        """生成以非终结符 symbol 的所有初始项目为 LR(0) 项目，以 lookahead 为展望符的所有 LR(1) 项目的集合"""
        return {self._create_item1(item0, lookahead) for item0 in self.symbol_to_start_item_list_hash[symbol]}

    def cal_concentric_hash(self) -> Dict[Tuple[int, ...], List[int]]:
        """计算 LR(1) 的项目集核心，并根据项目集的核心（仅包含规约符、符号列表和句柄的核心项目元组）进行聚合

        Returns
        -------
        Dict[Tuple[int, ...], List[Item1Set]]
            根据项目集核心聚合后的项目集
        """
        # 【性能设计】初始化方法中频繁使用的类属性，以避免重复获取类属性
        sid_to_core_tuple_hash: List[Tuple[int, ...]] = self.sid_to_core_tuple_hash

        concentric_hash = collections.defaultdict(list)
        for sid in self.sid_set:
            core_tuple = sid_to_core_tuple_hash[sid]
            centric_tuple = tuple(sorted(set(self.i1_id_to_i0_id_hash[i1_id] for i1_id in core_tuple)))
            # 根据项目集核心进行聚合
            concentric_hash[centric_tuple].append(sid)
        return concentric_hash

    def merge_same_concentric_item1_set(self) -> None:
        # pylint: disable=R0914
        """合并 LR(1) 项目集核心相同的 LR(1) 项目集（原地更新）"""

        # 【性能设计】初始化方法中频繁使用的类属性，以避免重复获取类属性
        core_tuple_to_sid_hash: Dict[Tuple[int, ...], int] = self.core_tuple_to_sid_hash
        sid_to_core_tuple_hash: List[Tuple[int, ...]] = self.sid_to_core_tuple_hash

        for sid_list in self.concentric_hash.values():
            if len(sid_list) == 1:
                continue  # 如果没有项目集核心相同的多个项目集，则不需要合并

            # 构造新的项目集
            new_core_item_set: Set[int] = set()  # 新项目集的核心项目
            new_other_item_set: Set[int] = set()  # 新项目集的其他等价项目
            for sid in sid_list:
                core_tuple = self.sid_to_core_tuple_hash[sid]
                all_i1_id_set = self.sid_to_other_i1_id_set_hash[sid]
                new_core_item_set |= set(core_tuple)
                new_other_item_set |= all_i1_id_set

            # 通过排序逻辑以保证结果状态是稳定的
            new_core_tuple = tuple(sorted(new_core_item_set))

            new_sid1 = len(self.sid_to_core_tuple_hash)
            core_tuple_to_sid_hash[new_core_tuple] = new_sid1
            sid_to_core_tuple_hash.append(new_core_tuple)
            self.sid_to_other_i1_id_set_hash.append(new_other_item_set)

            # 记录旧 core_tuple 到新 core_tuple 的映射
            for sid in sid_list:
                self.core_tuple_hash[sid] = new_sid1

            # 整理记录有效 SID1 的集合
            for sid in sid_list:
                if sid in self.sid_set:
                    self.sid_set.remove(sid)
            self.sid_set.add(new_sid1)

    def create_item1_set_relation(self) -> None:
        """构造 LR(1) 项目集之间的前驱 / 后继关系"""
        for sid1, successor_symbol, successor_sid1 in self.item1_set_relation:
            new_sid1 = self.core_tuple_hash.get(sid1, sid1)
            new_successor_sid1 = self.core_tuple_hash.get(successor_sid1, successor_sid1)
            self.s1_id_relation[new_sid1][successor_symbol] = new_successor_sid1

    def create_lr_parsing_table_use_lalr1(self) -> List[List[Callable]]:
        # pylint: disable=R0801
        # pylint: disable=R0912
        # pylint: disable=R0914
        """使用 LR(1) 解析器的逻辑，构造 LR_Parsing_Table

        pylint: disable=R0801 -- 未提高相似算法的代码可读性，允许不同算法之间存在少量相同代码

        Returns
        -------
        table : List[List[Callable]]
            ACTION 表 + GOTO 表
        """
        # 初始化 ACTION 二维表和 GOTO 二维表：第 1 维是状态 ID，第 2 维是符号 ID
        n_status = len(self.sid_set)
        table: List[List[Optional[Callable]]] = [[ActionError()] * self.grammar.n_symbol for _ in range(n_status)]

        position_shift_hash = {}  # ACTION + GOTO 表位置到移进操作列表的哈希映射（每个位置至多有一个 Shift 行为）
        position_reduce_list_hash = collections.defaultdict(list)  # ACTION + GOTO 表位置到规约操作列表的哈希映射（每个位置可以有多个 Reduce 行为）

        # 遍历所有项目集闭包，填充 ACTION 表和 GOTO 表（当前项目集即使是接收项目集，也需要填充）
        # 遍历所有有效 LR(1) 项目集闭包的 S1_ID
        for sid1 in self.sid_set:
            status_id = self.sid_to_status_hash[sid1]  # 计算 LR(1) 项目集的 S1_ID 对应的状态 ID

            # 根据项目集闭包的后继项目，填充 ACTION 表和 GOTO 表
            for successor_symbol, successor_s1_id in self.s1_id_relation[sid1].items():
                next_status_id = self.sid_to_status_hash[successor_s1_id]
                if self.grammar.is_terminal(successor_symbol):
                    # 后继项目为终结符，记录需要填充到 ACTION 表的 Shift 行为
                    position_shift_hash[(status_id, successor_symbol)] = ActionShift(status=next_status_id)
                else:
                    # 后继项目为非终结符，填充 GOTO 表
                    table[status_id][successor_symbol] = ActionGoto(status=next_status_id)

            # 遍历不包含后继项目的项目，记录需要填充到 ACTION 表的 Reduce 行为
            for i1_id in chain(self.sid_to_core_tuple_hash[sid1], self.sid_to_other_i1_id_set_hash[sid1]):
                i0_id = self.i1_id_to_i0_id_hash[i1_id]
                lookahead = self.i1_id_to_lookahead_hash[i1_id]
                item0 = self.i0_id_to_item0_hash[i0_id]
                if item0.successor_symbol is None:
                    reduce_action = ActionReduce(reduce_nonterminal_id=item0.nonterminal_id,
                                                 n_param=len(item0.before_handle),
                                                 reduce_function=item0.action)
                    position_reduce_list_hash[(status_id, lookahead)].append((
                        item0.rr_priority_idx,  # RR 优先级
                        item0.sr_priority_idx,  # SR 优先级
                        item0.sr_combine_type,  # SR 合并顺序
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
            shift_sr_priority_idx = self.grammar.get_terminal_sr_priority_idx(successor_symbol)  # 移进行为 SR 优先级
            shift_sr_combine_type = self.grammar.get_terminal_sr_combine_type(successor_symbol)  # 移进行为 SR 结合顺序

            # 获取规约行为信息
            _, reduce_sr_priority_idx, _, action_reduce = position_reduce_hash[position]

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

        # 当接受项目集闭包接收到结束符时，填充 Accept 行为
        table[self.accept_status_id][self.grammar.end_terminal] = ActionAccept()

        return table
