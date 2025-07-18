"""
LR(1) 文法解析器
"""

import cProfile
import collections
import dataclasses
import enum
from collections import defaultdict
from functools import lru_cache
from itertools import chain
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from metasequoia_parser.common import ActionAccept, ActionError, ActionGoto, ActionReduce, ActionShift
from metasequoia_parser.common import CombineType
from metasequoia_parser.common import Grammar
from metasequoia_parser.common import ParserBase
from metasequoia_parser.utils import LOGGER

EMPTY_SET = set()


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
    next_lr0_id : Optional[Item0]
        连接到的后继项目对象
    """

    # -------------------- 性能设计 --------------------
    # Item0 项目集唯一 ID
    # 通过在构造时添加 Item0 项目集的唯一 ID，从而将 Item0 项目集的哈希计算优化为直接获取唯一 ID
    id: int = dataclasses.field(kw_only=True, hash=True, compare=False)

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
    next_symbol: Optional[int] = dataclasses.field(kw_only=True, hash=False, compare=False)
    next_lr0_id: Optional[int] = dataclasses.field(kw_only=True, hash=False, compare=False)  # 连接到的后继项目对象

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

    def __init__(self,
                 grammar: Grammar,
                 debug: bool = False,
                 profile: bool = False,
                 trace_lr1: Optional[int] = None,
                 trace_reduce: Optional[Tuple[int, int]] = None,
                 trace_symbol_lookahead: Optional[Tuple[int, int]] = None,
                 trace_ah_lookahead: Optional[Tuple[int, int]] = None):
        """

        Parameters
        ----------
        grammar : Grammar
            文法对象
        debug : bool, default = False
            【调试】是否开启 Debug 模式日志
        profile : Optional[int], default = None
            【调试】如果不为 None 则开启步骤 4 的 cProfile 性能分析，且广度优先搜索的最大撒次数为 profile_4；如果为 None 则不开启性能分析
        trace_lr1 : Optional[int], default = None
            【调试】如果不为 None 则在编译过程中追踪 LR(1) 项目的来源，且 trace_lr1 为 LR(1) 项目 ID
        trace_reduce : Optional[Tuple[int, int]], default = None
            【调试】如果不为 None 则在编译过程中跟踪规约行为的来源，且 trace_reduce[0] 为状态 ID，trace_reduce[1] 为展望符 ID
        trace_symbol_lookahead : Optional[Tuple[int, int]], default = None
            【调试】如果不为 None 则在编译过程中跟踪符号和展望符组合的来源，且 trace_symbol_lookahead[0] 为符号 ID，trace_symbol_lookahead[1] 为展望符 ID
            符号和展望符的组合，表示该符号的初始项目为目标展望符的 LR(1) 项目组合
        trace_ah_lookahead : Optional[Tuple[int, int]], default = None
            【调试】如果不为 None 则在编译过程中跟踪句柄之后符号列表的 ID 和展望符组合的来源，且 trace_ah_lookahead[0] 为句柄之后符号列表的 ID，
            trace_ah_lookahead[1] 为展望符 ID
        """
        self.profile = profile
        self.grammar = grammar
        self.debug = debug

        # 【调试模式】追踪功能
        self._trace_lr1 = trace_lr1
        self._trace_reduce = trace_reduce
        self._trace_symbol_lookahead = trace_symbol_lookahead
        self._trace_ah_lookahead = trace_ah_lookahead

        # 【调试模式】cProfile 性能分析
        self.profiler = None
        if self.profile:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        # LR(0) 项目 ID 到 LR(0) 项目对象的映射
        self.lr0_list: List[Item0] = []

        # LR(1) 项目核心元组到 LR(1) 项目 ID 的映射
        # - LR(1) 项目核心元组包括指向的 LR(0) 项目 ID 和展望符
        self.lr1_core_to_lr1_id_hash: Dict[int, int] = {}

        LOGGER.info("[Step 1] START: 根据文法规则，构造所有的 LR(0) 项目及之间的关系")
        self.after_handle_to_ah_id_hash: Dict[Tuple[int, ...], int] = {}  # 句柄之后的符号列表到唯一 ID 的映射
        self.ah_id_to_after_handle_hash: List[Tuple[int, ...]] = []  # 唯一 ID 到句柄之后的符号列表的映射
        self.cal_all_lr0_list()
        LOGGER.info(f"[Step 1] END: "
                    f"LR(0) 项目数量 = {len(self.lr0_list)}, "
                    f"句柄之后的符号元组数量 = {len(self.after_handle_to_ah_id_hash)}")

        LOGGER.info("[Step 2] START: 根据所有 LR(0) 项目，构造每个终结符到其初始项目的映射")
        self.nonterminal_id_to_start_lr0_id_list_hash = self.cal_nonterminal_id_to_start_lr0_id_hash()
        LOGGER.info(f"[Step 2] END: "
                    f"映射表元素数量 = {len(self.nonterminal_id_to_start_lr0_id_list_hash)}")

        LOGGER.info("[Step 3] START: 根据所有 LR(0) 项目，获取初始 LR(0) 项目和接受 LR(0) 项目")
        self.init_lr0_id = self.get_init_lr0_id()
        self.accept_lr0_id = self.get_accept_lr0_id()
        LOGGER.info("[Step 3] END: "
                    f"初始 LR(0) 项目 ID = {self.init_lr0_id}, "
                    f"接受 LR(0) 项目 ID = {self.accept_lr0_id}")

        LOGGER.info("[Step 4] START: 根据所有 LR(0) 项目，广度优先搜索计算所有合并后的项目集闭包的核心 LR(0) 项目元组")
        self.closure_relation: List[Tuple[int, int, int, int]] = []  # LR(1) 项目集闭包之间的关联关系
        self.closure_relation_2: Dict[int, Dict[int]] = defaultdict(dict)
        self.closure_key_to_closure_id_hash = self.cal_core_to_item0_set_hash()
        LOGGER.info(f"[Step 4] END: "
                    f"合并后项目集闭包数量 = {len(self.closure_key_to_closure_id_hash)}")

        LOGGER.info("[Step 5] START: 根据合并后项目集闭包的数量，初始化 LR 分析表")
        # 初始化 LR 分析表（ACTION + GOTO）：第 1 维是状态 ID，第 2 维是符号 ID
        self.n_status = len(self.closure_key_to_closure_id_hash)
        self.lr_table: List[List[Optional[Callable]]] = [[ActionError()] * self.grammar.n_symbol
                                                         for _ in range(self.n_status)]
        self.lr_sr_priority: List[List[int]] = [[-1] * self.grammar.n_symbol for _ in range(self.n_status)]
        self.lr_rr_priority: List[List[int]] = [[-1] * self.grammar.n_symbol for _ in range(self.n_status)]
        LOGGER.info("[Step 5] END")

        LOGGER.info("[Step 6] START : 根据项目集闭包之间的前驱 / 后继关系，填写 LR 分析表中的 Action 行为和 Goto 行为")
        self.create_closure_relation()
        LOGGER.info("[Step 6] END")

        LOGGER.info("[Step 7] START: 根据文法规则，计算所有非终结符中可能的开头终结符")
        # 计算所有涉及的非终结符的符号 ID 的列表（之所以不使用语法中所有符号的列表，是因为部分符号可能没有被实际引用）
        nonterminal_id_list = list({lr0.nonterminal_id for lr0 in self.lr0_list})
        # 计算每个非终结符中，所有可能的开头终结符
        self.nonterminal_all_start_terminal = self.cal_nonterminal_all_start_terminal(nonterminal_id_list)
        LOGGER.info("[Step 7] END")

        # 根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射，同时构造项目集闭包之间的关联关系
        LOGGER.info(
            "[Step 8] START: 根据所有 LR(0) 项目、初始 LR(0) 项目、所有非终结符的开头终结符、每个终结符到其初始项目的映射、以及合并后项目集闭包的核心 LR(0) 项目元组")

        # LR(1) 项目 ID 到指向的 LR(0) 项目 ID 的映射
        self.lr1_id_to_lr0_id_hash: List[int] = []

        # LR(1) 项目 ID 到展望符的映射
        self.lr1_id_to_lookahead_hash: List[int] = []
        self.lr1_id_to_ah_id_and_lookahead_list: List[Tuple[int, int]] = []

        # LR(1) 项目 ID 到后继符号及后继 LR(1) 项目 ID
        self.lr1_id_to_next_symbol_next_lr1_id_hash: List[Tuple[int, int]] = []

        # 通过广度优先搜索，查找所有 LR(1) 项目集闭包及其之间的关联关系
        self.bfs_search_all_closure()

        LOGGER.info(f"[Step 8] END:"
                    f"LR(1) 项目数量 = {len(self.lr1_core_to_lr1_id_hash)}")

        LOGGER.info("[Step 9] START: 根据初始 LR(0) 项目和接受 LR(0) 项目，计算初始状态 ID 和接受状态 ID")
        self.init_status_id = self.closure_key_to_closure_id_hash[(self.init_lr0_id,)]
        accept_status_id = self.closure_key_to_closure_id_hash[(self.accept_lr0_id,)]
        LOGGER.info("[Step 9] END: "
                    f"初始状态 ID = {self.init_status_id}, "
                    f"接受状态 ID = {accept_status_id}")

        LOGGER.info("[Step 10] START: 根据接受状态 ID，构造 LR 分析表的 Accept 行为")
        self.lr_table[accept_status_id][self.grammar.end_terminal] = ActionAccept()
        self.table = self.lr_table
        LOGGER.info("[Step 10] END")

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

    def cal_all_lr0_list(self) -> None:
        """根据文法对象 Grammar 计算出所有项目（Item0 对象）的列表，并生成项目之间的后继关系

        Returns
        -------
        List[Item0]
            所有项目的列表
        """
        # 【性能设计】初始化方法中频繁使用的类属性，以避免重复获取类属性
        grammar: Grammar = self.grammar
        lr0_list: List[Item0] = self.lr0_list
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
                lr0_id = len(lr0_list)
                lr0_list.append(Item0(
                    id=lr0_id,
                    nonterminal_id=product.nonterminal_id,
                    before_handle=tuple(),
                    ah_id=0,
                    after_handle=tuple(),
                    action=product.action,
                    item_type=last_item_type,
                    next_symbol=None,  # 规约项目不存在后继项目
                    next_lr0_id=None,  # 规约项目不存在后继项目
                    sr_priority_idx=product.sr_priority_idx,
                    sr_combine_type=product.sr_combine_type,
                    rr_priority_idx=product.rr_priority_idx
                ))
                continue

            # 添加句柄在结束位置（最右侧）的项目（规约项目）
            lr0_id = len(lr0_list)
            last_lr0_id = lr0_id
            lr0_list.append(Item0(
                id=lr0_id,
                nonterminal_id=product.nonterminal_id,
                before_handle=tuple(product.symbol_id_list),
                ah_id=0,
                after_handle=tuple(),
                action=product.action,
                item_type=last_item_type,
                next_symbol=None,  # 规约项目不存在后继项目
                next_lr0_id=None,  # 规约项目不存在后继项目
                sr_priority_idx=product.sr_priority_idx,
                sr_combine_type=product.sr_combine_type,
                rr_priority_idx=product.rr_priority_idx
            ))

            # 从右向左依次添加句柄在中间位置（不是最左侧和最右侧）的项目（移进项目），并将上一个项目作为下一个项目的后继项目
            for i in range(len(product.symbol_id_list) - 1, 0, -1):
                lr0_id = len(lr0_list)
                after_handle = tuple(product.symbol_id_list[i:])
                if after_handle not in after_handle_to_ah_id_hash:
                    ah_id = len(ah_id_to_after_handle_hash)
                    after_handle_to_ah_id_hash[after_handle] = ah_id
                    ah_id_to_after_handle_hash.append(after_handle)
                else:
                    ah_id = after_handle_to_ah_id_hash[after_handle]
                lr0_list.append(Item0(
                    id=lr0_id,
                    nonterminal_id=product.nonterminal_id,
                    before_handle=tuple(product.symbol_id_list[:i]),
                    ah_id=ah_id,
                    after_handle=after_handle,
                    action=product.action,
                    item_type=ItemType.SHIFT,
                    next_symbol=product.symbol_id_list[i],
                    next_lr0_id=last_lr0_id,
                    sr_priority_idx=product.sr_priority_idx,
                    sr_combine_type=product.sr_combine_type,
                    rr_priority_idx=product.rr_priority_idx
                ))
                last_lr0_id = lr0_id

            # 添加添加句柄在开始位置（最左侧）的项目（移进项目或入口项目）
            lr0_id = len(lr0_list)
            after_handle = tuple(product.symbol_id_list)
            if after_handle not in after_handle_to_ah_id_hash:
                ah_id = len(ah_id_to_after_handle_hash)
                after_handle_to_ah_id_hash[after_handle] = ah_id
                ah_id_to_after_handle_hash.append(after_handle)
            else:
                ah_id = after_handle_to_ah_id_hash[after_handle]
            lr0_list.append(Item0(
                id=lr0_id,
                nonterminal_id=product.nonterminal_id,
                before_handle=tuple(),
                ah_id=ah_id,
                after_handle=after_handle,
                action=product.action,
                item_type=first_item_type,
                next_symbol=product.symbol_id_list[0],
                next_lr0_id=last_lr0_id,
                sr_priority_idx=product.sr_priority_idx,
                sr_combine_type=product.sr_combine_type,
                rr_priority_idx=product.rr_priority_idx
            ))

    def cal_nonterminal_id_to_start_lr0_id_hash(self) -> Dict[int, List[int]]:
        """根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表

        Returns
        -------
        Dict[int, List[T]]
            键为非终结符名称，值为非终结符对应项目的列表（泛型 T 为 ItemBase 的子类）
        """
        nonterminal_id_to_start_lr0_id_hash: Dict[int, List[int]] = collections.defaultdict(list)
        for lr0 in self.lr0_list:
            if not lr0.before_handle:
                nonterminal_id_to_start_lr0_id_hash[lr0.nonterminal_id].append(lr0.id)
        return nonterminal_id_to_start_lr0_id_hash

    def get_init_lr0_id(self) -> int:
        """从 LR(0) 项目列表中获取入口 LR(0) 项目的 ID"""
        for lr0 in self.lr0_list:
            if lr0.is_init():
                return lr0.id
        raise KeyError("未从项目列表中获取到 INIT 项目")

    def get_accept_lr0_id(self) -> int:
        """从 LR(0) 项目列表中获取入口 LR(0) 项目的 ID"""
        for lr0 in self.lr0_list:
            if lr0.is_accept():
                return lr0.id
        raise KeyError("未从项目列表中获取到 ACCEPT 项目")

    def create_lr1(self, lr0_id: int, lookahead: int) -> int:
        """如果 LR(1) 项目不存在则构造 LR(1) 项目对象，返回直接返回构造的 LR(1) 项目对象的 ID"""
        lr1_core = lr0_id * (self.grammar.n_terminal + 1) + lookahead
        if lr1_core in self.lr1_core_to_lr1_id_hash:
            return self.lr1_core_to_lr1_id_hash[lr1_core]
        lr0 = self.lr0_list[lr0_id]

        # 递归计算后继 LR(1) 项目
        if lr0.next_lr0_id is not None:
            next_lr1_id = self.create_lr1(lr0.next_lr0_id, lookahead)
        else:
            next_lr1_id = None

        # 初始化 LR(1) 项的基本信息映射
        lr1_id = len(self.lr1_core_to_lr1_id_hash)
        self.lr1_core_to_lr1_id_hash[lr1_core] = lr1_id
        self.lr1_id_to_lr0_id_hash.append(lr0_id)
        self.lr1_id_to_lookahead_hash.append(lookahead)

        self.lr1_id_to_ah_id_and_lookahead_list.append((lr0.ah_id, lookahead))
        self.lr1_id_to_next_symbol_next_lr1_id_hash.append((lr0.next_symbol, next_lr1_id))

        return lr1_id

    def cal_nonterminal_all_start_terminal(self, symbol_id_list: List[int]) -> Dict[int, Set[int]]:
        """计算每个非终结符中，所有可能的开头终结符

        Parameters
        ----------
        symbol_id_list : List[int]
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
            reduce_symbol = product.nonterminal_id
            for symbol in product.symbol_id_list:
                if grammar.is_terminal(symbol):
                    nonterminal_start_terminal[reduce_symbol].add(symbol)
                    break
                else:
                    nonterminal_start_nonterminal[reduce_symbol].add(symbol)

                # 如果当前符号为终结符，或为不允许匹配 %empty 的非终结符，则说明后续符号已不可能再包含开头字符
                if not grammar.is_maybe_empty(symbol):
                    break

        # 计算每个终结符直接或经过其他非终结符间接的开头终结符的列表
        nonterminal_all_start_terminal = collections.defaultdict(set)  # “非终结符名称” 到其 “直接或经由其他非终结符间接的开头终结符的列表” 的映射
        for nonterminal_name in symbol_id_list:
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

    def bfs_search_all_closure(self) -> None:
        """根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射，同时构造项目集闭包之间的关联关系"""

        # 【性能设计】初始化方法中频繁使用的类属性，以避免重复获取类属性
        lr1_id_to_next_symbol_next_lr1_id_hash = self.lr1_id_to_next_symbol_next_lr1_id_hash
        lr1_id_to_ah_id_and_lookahead_list = self.lr1_id_to_ah_id_and_lookahead_list
        n_terminal = self.grammar.n_terminal  # 【性能设计】提前获取需频繁使用的 grammar 中的常量，以减少调用次数

        # 根据入口项的 LR(0) 项构造 LR(1) 项
        init_lr1_id = self.create_lr1(self.init_lr0_id, self.grammar.end_terminal)

        # 初始化项目集闭包的广度优先搜索的队列：将入口项目集的核心项目元组添加到队列
        visited_2 = {(0, init_lr1_id)}
        visited_1 = {0}
        visited_3 = set()  # 已经访问的 closure_id 与 ah_id 的组合
        visited_4 = set()  # 已经访问的 closure_id、ah_id 与 lookahead 的组合
        visited_6 = set()  # 已经访问的 closure_id 和 lr1_id 的组合
        queue_1 = collections.deque([0])
        queue_2 = collections.defaultdict(list)
        queue_2[0].append(init_lr1_id)

        # 广度优先搜索遍历所有项目集闭包
        idx = 0
        while queue_1:
            closure_id = queue_1.popleft()
            visited_1.remove(closure_id)
            lr1_id_list = queue_2.pop(closure_id)
            lr1_id_tuple = tuple(lr1_id_list)

            idx += 1

            if self.debug is True and idx % 1000 == 0:
                LOGGER.info(f"正在广度优先搜索遍历所有项目集闭包: "
                            f"已处理={idx}, "
                            f"队列中={len(queue_2)}, "
                            f"LR(1) 项目数量={len(self.lr1_core_to_lr1_id_hash)}")

            for sub_lr1_id in lr1_id_tuple:
                next_symbol, next_lr1_id = lr1_id_to_next_symbol_next_lr1_id_hash[sub_lr1_id]
                if next_symbol is not None:
                    next_closure_id = self.closure_relation_2[closure_id][next_symbol]
                    # 将后继项目集闭包的核心项目元组添加到队列
                    if (next_closure_id, next_lr1_id) not in visited_2:
                        if next_closure_id not in visited_1:
                            visited_1.add(next_closure_id)
                            queue_1.append(next_closure_id)
                        visited_2.add((next_closure_id, next_lr1_id))
                        queue_2[next_closure_id].append(next_lr1_id)

                        # 【Debug】追踪 LR(1) 项目来源
                        if self._trace_lr1 is not None and next_lr1_id == self._trace_lr1:
                            lr0_id = self.lr1_id_to_lr0_id_hash[sub_lr1_id]
                            lookahead = self.lr1_id_to_lookahead_hash[sub_lr1_id]
                            lr0 = self.lr0_list[lr0_id]
                            print(f"LR(1) 项目来源位置 4: "
                                  f"last_lr1_id={sub_lr1_id}, lr0_id={lr0_id}, lr0={lr0}, lookahead={lookahead}")
                else:
                    self.add_lr1_to_closure(closure_id, sub_lr1_id)

            # 广度优先搜索，根据项目集核心项目元组（closure_core）生成项目集闭包中包含的其他项目列表（item_list）
            closure_other = set()

            # 初始化广度优先搜索的第 1 批节点
            combine_list = [lr1_id_to_ah_id_and_lookahead_list[lr1_id] for lr1_id in lr1_id_tuple]
            for ah_id, lookahead in combine_list:
                visited_4.add((closure_id, ah_id, lookahead))
            if self._trace_ah_lookahead is not None and self._trace_ah_lookahead in combine_list:
                for lr1_id in lr1_id_tuple:
                    ah_id, lookahead = lr1_id_to_ah_id_and_lookahead_list[lr1_id]
                    if self._trace_ah_lookahead != (ah_id, lookahead):
                        continue
                    lr0_id = self.lr1_id_to_lr0_id_hash[lr1_id]
                    lr0 = self.lr0_list[lr0_id]
                    symbol_name = self.grammar.get_symbol_name(lr0.nonterminal_id)
                    LOGGER.info(
                        f"[trace_ah_lookahead] 来源位置 2: "
                        f"symbol={lr0.nonterminal_id}({symbol_name}), lookahead={lookahead}, lr1_id={lr1_id},"
                        f"lr0={lr0}, lookahead={lookahead}")
            queue = collections.deque(combine_list)

            # ------------------------------ 广度优先搜索：计算 LR(1) 项目的项目集闭包【开始】 ------------------------------

            # 广度优先搜索所有的等价项目组
            while queue:
                sub_ah_id, sub_lookahead = queue.popleft()

                # 如果是规约项目，则一定不存在等价项目组，跳过该项目即可
                if sub_ah_id == 0:
                    continue

                after_handle = self.ah_id_to_after_handle_hash[sub_ah_id]

                # 获取当前句柄之后的第 1 个符号
                next_symbol = after_handle[0]

                # 如果当前句柄之后的第 1 个符号是终结符，则不存在等价的 LR(1) 项目，直接返回空集合
                # 【性能】通过 first_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
                if next_symbol < n_terminal:
                    continue

                combine_set, need_inherit = self.cal_generated_lookahead_set(sub_ah_id)
                if (self._trace_symbol_lookahead is not None and self._trace_symbol_lookahead in combine_set):
                    symbol_list = [(symbol, self.grammar.get_symbol_name(symbol)) for symbol in after_handle]
                    LOGGER.info(
                        f"[trace_symbol_lookahead] 组合来源位置 1: ah_id={sub_ah_id}, after_handle={[symbol_list]}")

                sub_lr1_id_set = set()
                next_closure_id = self.closure_relation_2[closure_id][next_symbol]

                # 如果 closure_id 和 sub_ah_id 相同，即当前状态和句柄后符号均相同，因为自生后继型 LR(1) 项目不依赖展望符，所以不需要重复处理
                if (closure_id, sub_ah_id) not in visited_3:
                    visited_3.add((closure_id, sub_ah_id))
                    sub_lr1_id_set |= combine_set

                if need_inherit is True:
                    if (self._trace_symbol_lookahead is not None
                            and next_symbol == self._trace_symbol_lookahead[0]
                            and sub_lookahead == self._trace_symbol_lookahead[1]):
                        LOGGER.info(f"[trace_symbol_lookahead] 组合来源位置 2: "
                                    f"closure_id={closure_id}, "
                                    f"ah_id={sub_ah_id}, "
                                    f"after_handle={self._debug_format_symbol_list(after_handle)}, "
                                    f"sub_lookahead={sub_lookahead}")
                    sub_lr1_id_set.add((next_symbol, sub_lookahead))

                # 将当前项目组匹配的等价项目组添加到所有等价项目组中
                closure_other |= sub_lr1_id_set

                # 将等价项目组中需要继续寻找等价项目的添加到队列
                # 【性能设计】在这里没有使用更 Pythonic 的批量操作，是因为批量操作会至少创建 2 个额外的集合，且会额外执行一次哈希计算，这带来的外性能消耗超过了 Python 循环和判断的额外消耗
                for symbol, lookahead in sub_lr1_id_set:
                    for sub_lr1_id in self.get_lr1_id_set_by_combine(symbol, lookahead):
                        sub_ah_id, sub_lookahead = lr1_id_to_ah_id_and_lookahead_list[sub_lr1_id]
                        if (closure_id, sub_ah_id, sub_lookahead) not in visited_4:
                            visited_4.add((closure_id, sub_ah_id, sub_lookahead))
                            queue.append((sub_ah_id, sub_lookahead))

                            if (self._trace_ah_lookahead is not None
                                    and sub_ah_id == self._trace_ah_lookahead[0]
                                    and sub_lookahead == self._trace_ah_lookahead[1]):
                                lr0_id = self.lr1_id_to_lr0_id_hash[sub_lr1_id]
                                lookahead = self.lr1_id_to_lookahead_hash[sub_lr1_id]
                                lr0 = self.lr0_list[lr0_id]
                                symbol_name = self.grammar.get_symbol_name(symbol)
                                LOGGER.info(
                                    f"[trace_ah_lookahead] 来源位置 1: symbol={symbol}({symbol_name}), lookahead={lookahead}, sub_lr1_id={sub_lr1_id},"
                                    f"lr0={lr0}, lookahead={lookahead}")

            # ------------------------------ 广度优先搜索：计算 LR(1) 项目的项目集闭包【结束】 ------------------------------

            # 根据后继项目符号进行分组，计算出每个后继项目集闭包的核心项目元组
            for symbol, lookahead in closure_other:
                for sub_lr1_id in self.get_lr1_id_set_by_combine(symbol, lookahead):

                    # 【Debug】
                    if self._trace_lr1 is not None and sub_lr1_id == self._trace_lr1:
                        symbol_name = self.grammar.get_symbol_name(symbol)
                        lookahead_name = self.grammar.get_symbol_name(lookahead)
                        LOGGER.info(f"[trace_lr1] LR(1) 项目来源位置 2: "
                                    f"closure_id={closure_id}, "
                                    f"symbol={symbol}({symbol_name}), "
                                    f"lookahead={lookahead}({lookahead_name})")

                    next_symbol, next_lr1_id = lr1_id_to_next_symbol_next_lr1_id_hash[sub_lr1_id]
                    if self._trace_lr1 is not None and next_lr1_id == self._trace_lr1:
                        LOGGER.info(f"[trace_lr1] 其他 LR(1) 的后继项目: closure_id={closure_id}, "
                                    f"{self._debug_format_lr1(sub_lr1_id)}")
                    if next_symbol is not None:
                        next_closure_id = self.closure_relation_2[closure_id][next_symbol]
                        # 将后继项目集闭包的核心项目元组添加到队列
                        if (next_closure_id, next_lr1_id) not in visited_2:
                            if next_closure_id not in visited_1:
                                visited_1.add(next_closure_id)
                                queue_1.append(next_closure_id)
                            visited_2.add((next_closure_id, next_lr1_id))
                            queue_2[next_closure_id].append(next_lr1_id)
                    else:
                        if (closure_id, sub_lr1_id) not in visited_6:
                            visited_6.add((closure_id, sub_lr1_id))
                            self.add_lr1_to_closure(closure_id, sub_lr1_id)

    def add_lr1_to_closure(self, closure_id: int, lr1_id: int) -> None:
        """将 LR(1) 项目添加到项目集闭包"""
        lr0_id = self.lr1_id_to_lr0_id_hash[lr1_id]
        lookahead = self.lr1_id_to_lookahead_hash[lr1_id]
        lr0 = self.lr0_list[lr0_id]
        if lr0.next_symbol is not None:
            return

        # 【Debug】跟踪规约行为的来源
        if (self._trace_reduce is not None
                and closure_id == self._trace_reduce[0]
                and lookahead == self._trace_reduce[1]):
            LOGGER.info(
                f"[trace_reduce] closure_id={closure_id}, lookahead={lookahead}({self.grammar.get_symbol_name(lookahead)}): "
                f"{self._debug_format_lr0(lr0_id)} lr1_id={lr1_id}")

        reduce_action = ActionReduce(reduce_nonterminal_id=lr0.nonterminal_id,
                                     n_param=len(lr0.before_handle),
                                     reduce_function=lr0.action)

        # 处理 规约/规约 冲突：如果 RR 优先级小于之前已经处理的规约行为，则不再处理当前规约行为
        if lr0.rr_priority_idx <= self.lr_rr_priority[closure_id][lookahead]:
            return
        self.lr_rr_priority[closure_id][lookahead] = lr0.rr_priority_idx  # 更新 RR 优先级

        # 处理 移进/规约 冲突：如果规约优先级大于移进优先级，则优先执行规约行为
        # 如果要规约的规则的 SR 优先级高于下一个输入符号的 SR 优先级，则进行规约
        if lr0.sr_priority_idx > self.lr_sr_priority[closure_id][lookahead]:
            self.lr_table[closure_id][lookahead] = reduce_action
            self.lr_sr_priority[closure_id][lookahead] = lr0.sr_priority_idx  # SR 优先级
        # 如果要规约的规则的 SR 优先级与下一个输入符号的 SR 优先级一致，即均使用同一级终结符的 SR 优先级，则根据该符号的结合方向计算移进行为 SR 结合顺序
        elif lr0.sr_priority_idx == self.lr_sr_priority[closure_id][lookahead]:
            shift_sr_combine_type = self.grammar.get_terminal_sr_combine_type(lookahead)
            if shift_sr_combine_type == CombineType.LEFT:
                # 如果结合方向为从左到右，则进行规约
                self.lr_table[closure_id][lookahead] = reduce_action
            elif shift_sr_combine_type != CombineType.RIGHT:
                # 如果既不是左结合也不是右结合，则抛出异常
                self.lr_table[closure_id][lookahead] = ActionError()
            # 如果结合方向为从右到左，则进行移进
        # 如果要规约的规则的 SR 优先级低于下一个输入符号的 SR 优先级，则进行移进（不需要进行额外处理）

    @lru_cache(maxsize=None)
    def cal_generated_lookahead_set(self, ah_id: int) -> Tuple[Set[Tuple[int, int]], bool]:
        n_terminal = self.grammar.n_terminal  # 【性能设计】提前获取需频繁使用的 grammar 中的常量，以减少调用次数
        after_handle = self.ah_id_to_after_handle_hash[ah_id]
        len_after_handle = len(after_handle)  # 【性能设计】提前计算需要频繁使用的常量
        first_symbol = after_handle[0]

        lookahead_set = set()  # 后继符的列表

        # 添加生成 symbol 非终结符对应的所有项目，并将这些项目也加入继续寻找等价项目组的队列中
        # 先从当前句柄后第 1 个元素向后继续遍历，添加自生型后继
        i = 1
        need_inherit = True  # 是否已经找到不能匹配 %empty 的非终结符或终结符
        while i < len_after_handle:  # 向后逐个遍历符号，寻找展望符
            next_symbol = after_handle[i]

            # 如果遍历到的符号是终结符，则将该终结符添加为展望符，则标记 is_stop 并结束遍历
            # 【性能】通过 next_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
            if next_symbol < n_terminal:
                lookahead_set.add(next_symbol)  # 自生后继符
                need_inherit = False
                break

            # 如果遍历到的符号是非终结符，则遍历该非终结符的所有可能的开头终结符添加为展望符
            for start_terminal in self.nonterminal_all_start_terminal[next_symbol]:
                lookahead_set.add(start_terminal)  # 自生后继符

            # 如果遍历到的非终结符不能匹配 %emtpy，则标记 is_stop 并结束遍历
            if not self.grammar.is_maybe_empty(next_symbol):
                need_inherit = False
                break

            i += 1

        return {(first_symbol, sub_lookahead) for sub_lookahead in lookahead_set}, need_inherit

    @lru_cache(maxsize=None)
    def get_lr1_id_set_by_combine(self, symbol: int, lookahead: int):
        lr1_id_set: Set[int] = set()  # 当前项目组之后的所有可能的 lookahead
        for lr0_id in self.nonterminal_id_to_start_lr0_id_list_hash[symbol]:
            lr1_id_set.add(self.create_lr1(lr0_id, lookahead))
        return lr1_id_set

    def create_closure_relation(self) -> None:
        """构造 LR(1) 项目集之间的前驱 / 后继关系"""
        n_terminal = self.grammar.n_terminal

        # 根据项目集闭包的后继项目，填充 ACTION 表和 GOTO 表
        for closure_id, next_symbol, next_closure_id, sr_priority_idx in self.closure_relation:
            if next_symbol < n_terminal:
                # 后继项目为终结符，记录需要填充到 ACTION 表的 Shift 行为
                self.lr_table[closure_id][next_symbol] = ActionShift(status=next_closure_id)
                self.lr_sr_priority[closure_id][next_symbol] = sr_priority_idx
            else:
                # 后继项目为非终结符，填充 GOTO 表
                self.lr_table[closure_id][next_symbol] = ActionGoto(status=next_closure_id)

    def cal_core_to_item0_set_hash(self) -> Dict[Tuple[int, ...], int]:
        """根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射但不构造项目集闭包之间的关联关系）

        Returns
        -------
        Dict[Item0, Item0Set]
            核心项目到项目集闭包的映射（项目集闭包中包含项目列表，但不包含项目闭包之间关联关系）
        """
        closure_relation = self.closure_relation

        # 将入口项目添加到广度优先搜索的队列中
        init_closure_core = (self.init_lr0_id,)
        visited = {init_closure_core}
        queue = collections.deque([(0, init_closure_core)])

        closure_key_to_closure_id_hash = {init_closure_core: 0}

        # 广度优先搜索遍历所有项目集闭包
        while queue:
            closure_id, closure_core = queue.popleft()

            # 根据 Item 生成项目集闭包中包含的项目列表
            closure_other = self.closure_lr0(closure_core)

            # 根据后继项目符号进行分组，计算出每个后继项目集闭包的核心项目元组
            next_group = collections.defaultdict(list)
            next_sr_priority = collections.Counter()
            for lr0_id in chain(closure_core, closure_other):
                lr0 = self.lr0_list[lr0_id]
                if len(lr0.after_handle) == 0:
                    continue  # 跳过匹配 %empty 的项目

                # 获取核心项目句柄之后的第一个符号
                next_symbol = lr0.after_handle[0]
                next_group[next_symbol].append(lr0.next_lr0_id)
                next_sr_priority[next_symbol] = max(next_sr_priority[next_symbol], lr0.sr_priority_idx)

            # 获取每个项目的作为后继项目集闭包的核心项目
            for next_symbol, sub_lr0_id_list in next_group.items():
                sr_priority_idx = next_sr_priority[next_symbol]
                next_closure_key: Tuple[int, ...] = tuple(sorted(set(sub_lr0_id_list)))
                if next_closure_key not in closure_key_to_closure_id_hash:
                    next_closure_id = len(closure_key_to_closure_id_hash)
                    closure_key_to_closure_id_hash[next_closure_key] = next_closure_id
                else:
                    next_closure_id = closure_key_to_closure_id_hash[next_closure_key]

                # 记录 LR(1) 项目集闭包之间的前驱 / 后继关系
                closure_relation.append((closure_id, next_symbol, next_closure_id, sr_priority_idx))
                self.closure_relation_2[closure_id][next_symbol] = next_closure_id

                # 将后继项目集闭包的核心项目元组添加到队列
                if next_closure_key not in visited:
                    queue.append((next_closure_id, next_closure_key))
                    visited.add(next_closure_key)

        return closure_key_to_closure_id_hash

    def closure_lr0(self, lr0_id_list: Tuple[int]) -> List[int]:
        """根据 Item0 生成项目集闭包（closure of item sets）中包含的项目列表

        Parameters
        ----------
        lr0_id_list : int
            项目集闭包的核心项目（最高层级项目）

        Returns
        -------
        List[int]
            项目集闭包中包含的 LR(0) 项目 ID 的列表
        """

        visited_symbol_set = set()  # 已访问过的句柄后第一个符号的集合
        queue = collections.deque()  # 待处理的句柄后第一个符号的集合

        for lr0_id in lr0_id_list:
            lr0 = self.lr0_list[lr0_id]
            if len(lr0.after_handle) == 0:
                continue  # 跳过匹配 %empty 的项目

            # 获取核心项目句柄之后的第一个符号
            first_symbol = lr0.after_handle[0]

            if first_symbol not in visited_symbol_set:
                visited_symbol_set.add(first_symbol)
                queue.append(first_symbol)

        # 初始化项目集闭包中包含的项目列表
        item_list = []

        # 广度优先搜索所有的等价项目组
        while queue:
            symbol = queue.popleft()

            # 如果当前符号是终结符，则不存在等价项目
            if symbol not in self.nonterminal_id_to_start_lr0_id_list_hash:
                continue

            # 添加生成 symbol 非终结符对应的所有项目，并将这些项目也加入继续寻找等价项目组的队列中
            for sub_lr0_id in self.nonterminal_id_to_start_lr0_id_list_hash[symbol]:
                item_list.append(sub_lr0_id)
                sub_lr0 = self.lr0_list[sub_lr0_id]

                if len(sub_lr0.after_handle) == 0:
                    continue  # 跳过匹配 %empty 的项目

                new_symbol = sub_lr0.after_handle[0]
                if new_symbol not in visited_symbol_set:
                    visited_symbol_set.add(new_symbol)
                    queue.append(new_symbol)

        return item_list

    def _debug_format_symbol_list(self, symbol_list: Iterable[int]) -> str:
        """【Debug】格式化符号 ID 的列表

        Parameters
        ----------
        symbol_list : Iterable[int]
            符号 ID 的列表
        """
        return "[" + ", ".join([f"{symbol}({self.grammar.get_symbol_name(symbol)})" for symbol in symbol_list]) + "]"

    def _debug_format_lr0(self, lr0_id: int) -> str:
        """【Debug】格式化 LR(0) 项目的 ID

        Parameters
        ----------
        lr0_id : int
            LR(0) 项目的 ID
        """
        lr0 = self.lr0_list[lr0_id]
        symbol = lr0.nonterminal_id
        symbol_name = self.grammar.get_symbol_name(symbol)
        before_handle = " ".join([f"{symbol}({self.grammar.get_symbol_name(symbol)})" for symbol in lr0.before_handle])
        after_handle = " ".join([f"{symbol}({self.grammar.get_symbol_name(symbol)})" for symbol in lr0.after_handle])
        return f"LR(0)={lr0_id}: {symbol}({symbol_name})->{before_handle}·{after_handle}"

    def _debug_format_lr1(self, lr1_id: int) -> str:
        """【Debug】格式化 LR(1) 项目的 ID

        Parameters
        ----------
        lr1_id : int
            LR(1) 项目的 ID
        """
        lr0_id = self.lr1_id_to_lr0_id_hash[lr1_id]
        lookahead = self.lr1_id_to_lookahead_hash[lr1_id]
        lookahead_name = self.grammar.get_symbol_name(lookahead)
        return f"LR(1)={lr1_id} detail: {self._debug_format_lr0(lr0_id)}, lookahead={lookahead}({lookahead_name})"
