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
        self.closure_relation: List[Tuple[int, int, int]] = []  # LR(1) 项目集闭包之间的关联关系
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
        self.closure_id_to_closure_set_hash: List[Set[int]] = [set() for _ in range(self.n_status)]

        # LR(1) 项目 ID 到指向的 LR(0) 项目 ID 的映射
        self.lr1_id_to_lr0_id_hash: List[int] = []

        # 查询组合 ID 到句柄之后的符号列表的唯一 ID 与展望符的组合的唯一 ID 的映射
        self.cid_to_ah_id_and_lookahead_list: List[Tuple[int, int]] = []
        self.ah_id_and_lookahead_to_cid_hash: Dict[Tuple[int, int], int] = {}  # 用于构造唯一 ID

        # LR(1) 项目 ID 到组合 ID 的映射
        self.lr1_id_to_cid_hash: List[int] = []
        self.ah_id_no_lookahead_to_cid_hash: Dict[int, int] = {}

        # LR(1) 项目 ID 到展望符的映射
        self.lr1_id_to_lookahead_hash: List[int] = []

        # LR(1) 项目 ID 到后继符号及后继 LR(1) 项目 ID
        self.lr1_id_to_next_symbol_next_lr1_id_hash: List[Tuple[int, int]] = []

        self.init_lr1_id: Optional[int] = None

        # 通过广度优先搜索，查找所有 LR(1) 项目集闭包及其之间的关联关系
        self.bfs_search_all_closure()

        LOGGER.info(f"[Step 8] END:"
                    f"LR(1) 项目数量 = {len(self.lr1_core_to_lr1_id_hash)}")

        LOGGER.info("[Step 9] START: 根据初始 LR(0) 项目和接受 LR(0) 项目，计算初始状态 ID 和接受状态 ID")
        init_closure_core = (self.init_lr1_id,)
        init_closure_key = self.get_closure_key_by_clsure_core(init_closure_core)
        accept_lr1_core = self.accept_lr0_id * (self.grammar.n_terminal + 1) + self.grammar.end_terminal
        accept_lr1_id = self.lr1_core_to_lr1_id_hash[accept_lr1_core]
        accept_closure_core = (accept_lr1_id,)
        accept_closure_key = self.get_closure_key_by_clsure_core(accept_closure_core)
        self.init_status_id = self.closure_key_to_closure_id_hash[init_closure_key]
        self.accept_status_id = self.closure_key_to_closure_id_hash[accept_closure_key]
        LOGGER.info("[Step 9] END: "
                    f"初始状态 ID = {self.init_status_id}, "
                    f"接受状态 ID = {self.accept_status_id}")

        LOGGER.info("[Step 10] START: 根据接受状态 ID，构造 LR 分析表的 Accept 行为")
        self.lr_table[self.accept_status_id][self.grammar.end_terminal] = ActionAccept()
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

        ah_id = lr0.ah_id
        self.lr1_id_to_cid_hash.append(self.create_ah_id_lookahead_combine(ah_id, lookahead))

        # 添加 lookahead 为空的 cid
        self.ah_id_no_lookahead_to_cid_hash[ah_id] = self.create_ah_id_lookahead_combine(ah_id, self.grammar.n_terminal)

        self.lr1_id_to_next_symbol_next_lr1_id_hash.append((lr0.next_symbol, next_lr1_id))

        return lr1_id

    def create_ah_id_lookahead_combine(self, ah_id: int, lookahead: int) -> int:
        if (ah_id, lookahead) in self.ah_id_and_lookahead_to_cid_hash:
            return self.ah_id_and_lookahead_to_cid_hash[(ah_id, lookahead)]
        cid = len(self.ah_id_and_lookahead_to_cid_hash)
        self.ah_id_and_lookahead_to_cid_hash[(ah_id, lookahead)] = cid
        self.cid_to_ah_id_and_lookahead_list.append((ah_id, lookahead))
        return cid

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
        closure_key_to_closure_id_hash = self.closure_key_to_closure_id_hash
        closure_id_to_closure_set_hash = self.closure_id_to_closure_set_hash
        lr1_id_to_next_symbol_next_lr1_id_hash = self.lr1_id_to_next_symbol_next_lr1_id_hash

        # 根据入口项的 LR(0) 项构造 LR(1) 项
        self.init_lr1_id = self.create_lr1(self.init_lr0_id, self.grammar.end_terminal)
        init_closure_core = (self.init_lr1_id,)

        # 初始化项目集闭包的广度优先搜索的队列：将入口项目集的核心项目元组添加到队列
        visited = {init_closure_core}
        queue = collections.deque([(0, init_closure_core)])

        # 广度优先搜索遍历所有项目集闭包
        idx = 0
        while queue:
            closure_id, closure_core = queue.popleft()

            idx += 1

            if self.debug is True and idx % 1000 == 0:
                LOGGER.info(f"正在广度优先搜索遍历所有项目集闭包: "
                            f"已处理={idx}, "
                            f"队列中={len(queue)}, "
                            f"LR(1) 项目数量={len(self.lr1_core_to_lr1_id_hash)}, "
                            f"LR(1) 项目集闭包数量={len(self.closure_id_to_closure_set_hash)}")

            # 广度优先搜索，根据项目集核心项目元组（closure_core）生成项目集闭包中包含的其他项目列表（item_list）
            closure_other = self.new_closure_lr1(tuple(sorted(closure_core)))
            # print(f"项目集闭包尺寸: {len(closure_core)}, {len(other_lr1_id_set)}({sys.getsizeof(other_lr1_id_set)})")

            # 构造项目集闭包并添加到结果集中
            closure_id_to_closure_set_hash[closure_id] |= set(closure_core)
            closure_id_to_closure_set_hash[closure_id] |= closure_other

            for lr1_id in chain(closure_core, closure_other):
                lr0_id = self.lr1_id_to_lr0_id_hash[lr1_id]
                lookahead = self.lr1_id_to_lookahead_hash[lr1_id]
                lr0 = self.lr0_list[lr0_id]
                if lr0.next_symbol is None:
                    reduce_action = ActionReduce(reduce_nonterminal_id=lr0.nonterminal_id,
                                                 n_param=len(lr0.before_handle),
                                                 reduce_function=lr0.action)

                    # 处理 规约/规约 冲突：如果 RR 优先级小于之前已经处理的规约行为，则不再处理当前规约行为
                    if lr0.rr_priority_idx <= self.lr_rr_priority[closure_id][lookahead]:
                        continue
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

            # 根据后继项目符号进行分组，计算出每个后继项目集闭包的核心项目元组
            next_group = collections.defaultdict(list)
            for lr1_id in chain(closure_core, closure_other):
                next_symbol, next_lr1_id = lr1_id_to_next_symbol_next_lr1_id_hash[lr1_id]
                if next_symbol is not None:
                    next_group[next_symbol].append(next_lr1_id)

            # 计算后继项目集的核心项目元组（排序以保证顺序稳定）
            for next_symbol, sub_lr1_id_set in next_group.items():
                next_closure_core: Tuple[int, ...] = tuple(sorted(set(sub_lr1_id_set)))
                next_closure_key: Tuple[int, ...] = self.get_closure_key_by_clsure_core(next_closure_core)
                assert next_closure_key in closure_key_to_closure_id_hash, f"{next_closure_key} 未被 LR(0) 计算"
                next_closure_id = closure_key_to_closure_id_hash[next_closure_key]

                # 将后继项目集闭包的核心项目元组添加到队列
                if next_closure_core not in visited:
                    queue.append((next_closure_id, next_closure_core))
                    visited.add(next_closure_core)

    def new_closure_lr1(self, closure_core: Tuple[int]) -> Set[int]:
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
        cid_to_ah_id_and_lookahead_list = self.cid_to_ah_id_and_lookahead_list
        lr1_id_to_cid_hash = self.lr1_id_to_cid_hash

        lr1_id_set = set()
        visited_ah_id_set = set()  # 已经处理的 ah_id 的集合
        for lr1_id in closure_core:
            cid = lr1_id_to_cid_hash[lr1_id]
            ah_id, lookahead = cid_to_ah_id_and_lookahead_list[cid]
            if ah_id == 0:
                continue
            sub_lr1_id_set, lr0_id_set = self.cal_generated_and_inherit_lr1_id_set_by_ah_id(ah_id)

            # 对于自发后继型 LR(1) 项目，每个 ah_id 只需要处理一次
            if ah_id not in visited_ah_id_set:
                lr1_id_set |= sub_lr1_id_set
                visited_ah_id_set.add(ah_id)

            for lr0_id in lr0_id_set:
                lr1_id_set.add(self.create_lr1(lr0_id, lookahead))
        return lr1_id_set

    def compute_single_level_closure(self, cid: int) -> Set[int]:
        """计算 LR(1) 单层的等价 LR(1) 项目的 ID 的集合

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
            return EMPTY_SET

        n_terminal = self.grammar.n_terminal  # 【性能设计】提前获取需频繁使用的 grammar 中的常量，以减少调用次数
        after_handle = self.ah_id_to_after_handle_hash[ah_id]

        # 获取当前句柄之后的第 1 个符号
        first_symbol = after_handle[0]

        # 如果当前句柄之后的第 1 个符号是终结符，则不存在等价的 LR(1) 项目，直接返回空集合
        # 【性能】通过 first_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
        if first_symbol < n_terminal:
            return EMPTY_SET

        lr1_id_set, need_inherit = self.cal_generated_lookahead_set(ah_id)

        if need_inherit is True:
            return lr1_id_set | {self.create_lr1(lr0_id, lookahead)
                                 for lr0_id in self.nonterminal_id_to_start_lr0_id_list_hash[first_symbol]}
        else:
            return lr1_id_set

    @lru_cache(maxsize=None)
    def cal_generated_lookahead_set(self, ah_id: int) -> Tuple[Set[int], bool]:
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

        lr1_id_set: Set[int] = set()  # 当前项目组之后的所有可能的 lookahead
        for lr0_id in self.nonterminal_id_to_start_lr0_id_list_hash[first_symbol]:
            for sub_lookahead in lookahead_set:
                lr1_id_set.add(self.create_lr1(lr0_id, sub_lookahead))

        return lr1_id_set, need_inherit

    @lru_cache(maxsize=None)
    def cal_generated_and_inherit_lr1_id_set_by_ah_id(self, ah_id: int) -> Tuple[Set[int], Set[int]]:
        # 初始化广度优先搜索的第 1 批节点
        lr1_id_to_cid_hash = self.lr1_id_to_cid_hash
        lr1_id_to_lr0_id_hash = self.lr1_id_to_lr0_id_hash
        lr1_id_to_lookahead_hash = self.lr1_id_to_lookahead_hash

        cid = self.ah_id_no_lookahead_to_cid_hash[ah_id]
        visited_cid_set = {cid}
        queue = collections.deque([cid])

        # 广度优先搜索所有的等价项目组
        lr1_id_set = set()
        while queue:
            # 计算单层的等价 LR(1) 项目
            sub_lr1_id_set = self.compute_single_level_closure(queue.popleft())

            # 将当前项目组匹配的等价项目组添加到所有等价项目组中
            lr1_id_set |= sub_lr1_id_set

            # 将等价项目组中需要继续寻找等价项目的添加到队列
            # 【性能设计】在这里没有使用更 Pythonic 的批量操作，是因为批量操作会至少创建 2 个额外的集合，且会额外执行一次哈希计算，这带来的外性能消耗超过了 Python 循环和判断的额外消耗
            for lr1_id in sub_lr1_id_set:
                new_cid = lr1_id_to_cid_hash[lr1_id]
                if new_cid not in visited_cid_set:
                    visited_cid_set.add(new_cid)
                    queue.append(new_cid)

        generated_lr1_id_set = set()  # 自生后继型 LR(1) 项目
        inherit_lr0_id_set = set()  # 继承后继型 LR(1) 项目对应的 LR(0) 项目
        for lr1_id in lr1_id_set:
            if lr1_id_to_lookahead_hash[lr1_id] != self.grammar.n_terminal:
                generated_lr1_id_set.add(lr1_id)
            else:
                inherit_lr0_id_set.add(lr1_id_to_lr0_id_hash[lr1_id])

        return generated_lr1_id_set, inherit_lr0_id_set

    def get_closure_key_by_clsure_core(self, closure_core: Tuple[int, ...]) -> Tuple[int, ...]:
        """根据 LR(1) 项目集闭包的 LR(1) 项目的元组，计算 LR(1) 项目集闭包用于合并的 LR(0) 项目的元组"""
        lr1_id_to_lr0_id_hash = self.lr1_id_to_lr0_id_hash
        return tuple(sorted(set(lr1_id_to_lr0_id_hash[lr1_id] for lr1_id in closure_core)))

    def create_closure_relation(self) -> None:
        """构造 LR(1) 项目集之间的前驱 / 后继关系"""
        n_terminal = self.grammar.n_terminal

        # 根据项目集闭包的后继项目，填充 ACTION 表和 GOTO 表
        for closure_id, next_symbol, next_closure_id in self.closure_relation:
            if next_symbol < n_terminal:
                # 后继项目为终结符，记录需要填充到 ACTION 表的 Shift 行为
                self.lr_table[closure_id][next_symbol] = ActionShift(status=next_closure_id)
                self.lr_sr_priority[closure_id][next_symbol] = self.grammar.get_terminal_sr_priority_idx(next_symbol)
            else:
                # 后继项目为非终结符，填充 GOTO 表
                self.lr_table[closure_id][next_symbol] = ActionGoto(status=next_closure_id)

    def create_lr_parsing_table_use_lalr1(self) -> None:
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
        # closure_id_to_closure_set_hash = self.closure_id_to_closure_set_hash
        # lr1_id_to_lr0_id_hash = self.lr1_id_to_lr0_id_hash
        # lr1_id_to_lookahead_hash = self.lr1_id_to_lookahead_hash
        # lr0_list = self.lr0_list
        #
        # # 遍历所有项目集闭包，填充 ACTION 表和 GOTO 表（当前项目集即使是接收项目集，也需要填充）
        # # 遍历所有有效 LR(1) 项目集闭包的 S1_ID
        # for closure_id in range(self.n_status):
        #     # 遍历不包含后继项目的项目，记录需要填充到 ACTION 表的 Reduce 行为
        #     for lr1_id in closure_id_to_closure_set_hash[closure_id]:
        #         lr0_id = lr1_id_to_lr0_id_hash[lr1_id]
        #         lookahead = lr1_id_to_lookahead_hash[lr1_id]
        #         lr0 = lr0_list[lr0_id]
        #         if lr0.next_symbol is None:
        #             reduce_action = ActionReduce(reduce_nonterminal_id=lr0.nonterminal_id,
        #                                          n_param=len(lr0.before_handle),
        #                                          reduce_function=lr0.action)
        #
        #             # 处理 规约/规约 冲突：如果 RR 优先级小于之前已经处理的规约行为，则不再处理当前规约行为
        #             if lr0.rr_priority_idx <= self.lr_rr_priority[closure_id][lookahead]:
        #                 continue
        #             self.lr_rr_priority[closure_id][lookahead] = lr0.rr_priority_idx  # 更新 RR 优先级
        #
        #             # 处理 移进/规约 冲突：如果规约优先级大于移进优先级，则优先执行规约行为
        #             # 如果要规约的规则的 SR 优先级高于下一个输入符号的 SR 优先级，则进行规约
        #             if lr0.sr_priority_idx > self.lr_sr_priority[closure_id][lookahead]:
        #                 self.lr_table[closure_id][lookahead] = reduce_action
        #                 self.lr_sr_priority[closure_id][lookahead] = lr0.sr_priority_idx  # SR 优先级
        #             # 如果要规约的规则的 SR 优先级与下一个输入符号的 SR 优先级一致，即均使用同一级终结符的 SR 优先级，则根据该符号的结合方向计算移进行为 SR 结合顺序
        #             elif lr0.sr_priority_idx == self.lr_sr_priority[closure_id][lookahead]:
        #                 shift_sr_combine_type = self.grammar.get_terminal_sr_combine_type(lookahead)
        #                 if shift_sr_combine_type == CombineType.LEFT:
        #                     # 如果结合方向为从左到右，则进行规约
        #                     self.lr_table[closure_id][lookahead] = reduce_action
        #                 elif shift_sr_combine_type != CombineType.RIGHT:
        #                     # 如果既不是左结合也不是右结合，则抛出异常
        #                     self.lr_table[closure_id][lookahead] = ActionError()
        #                 # 如果结合方向为从右到左，则进行移进
        #             # 如果要规约的规则的 SR 优先级低于下一个输入符号的 SR 优先级，则进行移进（不需要进行额外处理）

        # 当接受项目集闭包接收到结束符时，填充 Accept 行为


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
            for lr0_id in chain(closure_core, closure_other):
                lr0 = self.lr0_list[lr0_id]
                if len(lr0.after_handle) == 0:
                    continue  # 跳过匹配 %empty 的项目

                # 获取核心项目句柄之后的第一个符号
                next_symbol = lr0.after_handle[0]
                next_group[next_symbol].append(lr0.next_lr0_id)

            # 获取每个项目的作为后继项目集闭包的核心项目
            for next_symbol, sub_lr0_id_list in next_group.items():
                next_closure_key: Tuple[int, ...] = tuple(sorted(set(sub_lr0_id_list)))
                if next_closure_key not in closure_key_to_closure_id_hash:
                    next_closure_id = len(closure_key_to_closure_id_hash)
                    closure_key_to_closure_id_hash[next_closure_key] = next_closure_id
                else:
                    next_closure_id = closure_key_to_closure_id_hash[next_closure_key]

                # 记录 LR(1) 项目集闭包之间的前驱 / 后继关系
                closure_relation.append((closure_id, next_symbol, next_closure_id))

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
