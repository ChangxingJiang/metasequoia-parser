"""
LR(1) 文法解析器
"""

import abc
import cProfile
import collections
import dataclasses
import enum
from functools import lru_cache
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
class ItemBase(abc.ABC):
    # pylint: disable=R0902
    """项目类的抽象基类"""

    @abc.abstractmethod
    def __repr__(self) -> str:
        """将 ItemBase 转换为字符串表示"""

    @abc.abstractmethod
    def get_symbol_id(self) -> int:
        """获取符号 ID"""

    @abc.abstractmethod
    def get_before_handle(self) -> Tuple[int, ...]:
        """获取句柄之前的符号名称的列表"""

    @abc.abstractmethod
    def get_after_handle(self) -> Tuple[int, ...]:
        """获取句柄之后的符号名称的列表"""

    @abc.abstractmethod
    def is_init(self) -> bool:
        """是否为入口项目"""

    @abc.abstractmethod
    def is_accept(self) -> bool:
        """是否为接受项目"""


@dataclasses.dataclass(slots=True, frozen=True, eq=True, order=True)
class Item0(ItemBase):
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
    after_handle: Tuple[int, ...] = dataclasses.field(kw_only=True, hash=False, compare=True)  # 在句柄之后的符号名称的列表
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

    def get_symbol_id(self) -> int:
        """获取符号 ID"""
        return self.nonterminal_id

    def get_before_handle(self) -> Tuple[int, ...]:
        """获取句柄之前的符号名称的列表"""
        return self.before_handle

    def get_after_handle(self) -> Tuple[int, ...]:
        """获取句柄之后的符号名称的列表"""
        return self.after_handle

    def is_init(self) -> bool:
        """是否为入口项目"""
        return self.item_type == ItemType.INIT

    def is_accept(self) -> bool:
        """是否为接收项目"""
        return self.item_type == ItemType.ACCEPT


@dataclasses.dataclass(slots=True, frozen=True, eq=True, order=True)
class Item1(ItemBase):
    """提前查看下一个字符的项目类：适用于 LR(1) 解析器和 LALR(1) 解析器

    Attributes
    ----------
    successor_item : Optional[Item0]
        连接到的后继项目对象
    """

    # -------------------- 性能设计 --------------------
    # 【性能设计】Item1 项目集唯一 ID
    # 通过在构造时添加 Item1 项目集的唯一 ID，从而将 Item1 项目集的哈希计算优化为直接获取唯一 ID
    id: int = dataclasses.field(kw_only=True, hash=True, compare=False)

    # -------------------- 项目的基本属性 --------------------
    item0: Item0 = dataclasses.field(kw_only=True, hash=False, compare=True)  # 连接到的后继项目对象
    lookahead: int = dataclasses.field(kw_only=True, hash=False, compare=True)  # 展望符（终结符）
    successor_item: Optional[int] = dataclasses.field(kw_only=True, hash=False, compare=False)  # 指向后继 LR(1) 项目的指针

    @staticmethod
    def create_by_item0(i1_id: int, item0: Item0, lookahead: int, successor_item1: Optional["Item1"]) -> "Item1":
        """采用享元模式，通过 Item0 对象和 lookahead 构造 Item1 对象

        Parameters
        ----------
        item0 : Item0
            构造的 Item0 文法项目对象
        lookahead : int
            展望符

        Returns
        -------
        Item1
            构造的 Item1 文法项目对象
        """
        item1 = Item1(
            id=i1_id,
            item0=item0,
            successor_item=successor_item1,
            lookahead=lookahead,
        )
        return item1

    def __repr__(self) -> str:
        """将 ItemBase 转换为字符串表示"""
        return f"{self.item0},{self.lookahead}"

    def get_symbol_id(self) -> int:
        """获取符号 ID"""
        return self.item0.nonterminal_id

    def get_before_handle(self) -> Tuple[int, ...]:
        """获取句柄之前的符号名称的列表"""
        return self.item0.before_handle

    def get_after_handle(self) -> Tuple[int, ...]:
        """获取句柄之后的符号名称的列表"""
        return self.item0.after_handle

    def is_init(self) -> bool:
        """是否为入口项目"""
        return self.item0.item_type == ItemType.INIT

    def is_accept(self) -> bool:
        """是否为接收项目"""
        return self.item0.item_type == ItemType.ACCEPT


@dataclasses.dataclass(slots=True)
class Item1Set:
    """提前查看下一个字符的项目集闭包类：适用于 LR(1) 解析器和 LALR(1) 解析器"""

    sid: int = dataclasses.field(kw_only=True)
    core_tuple: Tuple[int, ...] = dataclasses.field(kw_only=True)  # 核心项目
    other_item1_set: Set[int] = dataclasses.field(kw_only=True)  # 项目集闭包中除核心项目外的其他等价项目

    @staticmethod
    def create(sid: int, core_list: Tuple[int, ...], other_item1_set: Set[int]) -> "Item1Set":
        """项目集闭包对象的构造方法"""
        return Item1Set(
            sid=sid,
            core_tuple=core_list,
            other_item1_set=other_item1_set
        )

    def __repr__(self):
        core_tuple_str = "|".join(str(item) for item in self.core_tuple)
        return f"[{core_tuple_str}]"


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

        # LR(1) 项目 ID 到 LR(1) 项目对象的映射
        self.i1_id_to_item1_hash: List[Item1] = []

        # 根据文法计算所有项目（Item0 对象），并生成项目之间的后继关系
        LOGGER.info("[1 / 10] 计算 Item0 对象开始")
        self.after_handle_to_ah_id_hash: Dict[Tuple[int, ...], int] = {}  # 句柄之后的符号列表到唯一 ID 的映射
        self.ah_id_to_after_handle_hash: List[Tuple[int, ...]] = []  # 唯一 ID 到句柄之后的符号列表的映射
        self.cal_all_item0_list()
        LOGGER.info(f"[1 / 10] 计算 Item0 对象结束 (Item0 对象数量 = {len(self.i0_id_to_item0_hash)})")

        # 根据所有项目的列表，构造每个非终结符到其初始项目（句柄在最左侧）列表的映射表
        LOGGER.info("[2 / 10] 构造非终结符到其初始项目列表的映射表开始")
        self.symbol_to_start_item_list_hash = self.cal_symbol_to_start_item_list_hash()
        LOGGER.info(f"[2 / 10] 构造非终结符到其初始项目列表的映射表结束 "
                    f"(映射表元素数量 = {len(self.symbol_to_start_item_list_hash)})")

        # 从项目列表中获取入口项目
        LOGGER.info("[3 / 10] 从项目列表中获取入口项目开始")
        self.init_item0 = self.cal_init_item_from_item_list()
        LOGGER.info("[3 / 10] 从项目列表中获取入口项目结束")

        # 计算所有非终结符名称的列表
        nonterminal_name_list = list({item0.nonterminal_id for item0 in self.i0_id_to_item0_hash})

        # 计算每个非终结符中，所有可能的开头终结符
        self.nonterminal_all_start_terminal = self.cal_nonterminal_all_start_terminal(nonterminal_name_list)

        # 根据入口项目以及非标识符对应开始项目的列表，使用广度优先搜索，构造所有核心项目到项目集闭包的映射，同时构造项目集闭包之间的关联关系
        LOGGER.info("[4 / 10] 广度优先搜索，构造项目集闭包之间的关联关系")
        self.item1_set_relation: List[Tuple[int, int, int]] = []  # 核心项目之间的关联关系
        self.core_tuple_to_sid_hash: Dict[Tuple[int, ...], int] = {}  # 核心项目到 SID1 的映射
        self.sid_to_core_tuple_hash: List[Tuple[int, ...]] = []  # SID1 到核心项目的映射
        self.sid_to_item1_set_hash: List[Item1Set] = []  # SID1 到 LR(1) 项目集的映射

        # 广度优先搜索，查找 LR(1) 项目集及之间的关联关系
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
        LOGGER.info("[9 / 10] 生成初始状态开始")
        self.init_item1 = self._create_item1(self.init_item0, self.grammar.end_terminal)
        self.entrance_status_id = self.sid_to_status_hash[self.core_tuple_to_sid_hash[(self.init_item1,)]]
        LOGGER.info("[9 / 10] 生成初始状态结束")

        # 构造 ACTION 表 + GOTO 表
        LOGGER.info("[10 / 10] 构造 ACTION 表 + GOTO 表开始")
        # 计算接受 LR(1) 项目集对应的状态 ID
        self.accept_s1_id = None
        for s1_id, core_tuple in enumerate(self.sid_to_core_tuple_hash):
            for i1_id in core_tuple:
                item1 = self.i1_id_to_item1_hash[i1_id]
                if item1.item0.is_accept():
                    self.accept_s1_id = s1_id
                    break
        self.accept_status_id = self.sid_to_status_hash[self.accept_s1_id]

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
        return self.table, self.entrance_status_id

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
                now_item = Item0(
                    i0_id=i0_id,
                    nonterminal_id=product.nonterminal_id,
                    before_handle=tuple(product.symbol_id_list[:i]),
                    after_handle=tuple(product.symbol_id_list[i:]),
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
            i0_id_to_item0_hash.append(Item0(
                i0_id=i0_id,
                nonterminal_id=product.nonterminal_id,
                before_handle=tuple(),
                after_handle=tuple(product.symbol_id_list),
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
        for item in self.i0_id_to_item0_hash:
            if len(item.get_before_handle()) == 0:
                symbol_to_start_item_list_hash[item.get_symbol_id()].append(item)
        return symbol_to_start_item_list_hash

    def cal_init_item_from_item_list(self) -> Item0:
        """从项目列表中获取入口项目"""
        for item0 in self.i0_id_to_item0_hash:
            if item0.is_init():
                return item0
        raise KeyError("未从项目列表中获取到 INIT 项目")

    def _create_item1(self, item0: Item0, lookahead: int) -> int:
        """如果 item1 不存在则构造 item1，返回直接返回已构造的 item1"""
        # 如果 item1 已经存在则返回已存在 item1
        if (item0.i0_id, lookahead) in self.item1_core_to_i1_id_hash:
            return self.item1_core_to_i1_id_hash[(item0.i0_id, lookahead)]

        if item0.successor_item is not None:
            successor_item1 = self._create_item1(item0.successor_item, lookahead)
        else:
            successor_item1 = None

        i1_id = len(self.i1_id_to_item1_hash)
        item1 = Item1.create_by_item0(
            i1_id=i1_id,
            item0=item0,
            lookahead=lookahead,
            successor_item1=successor_item1
        )
        self.item1_core_to_i1_id_hash[(item0.i0_id, lookahead)] = i1_id
        self.i1_id_to_item1_hash.append(item1)
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
        core_tuple_to_sid_hash: Dict[Tuple[int, ...], int] = self.core_tuple_to_sid_hash
        sid_to_core_tuple_hash: List[Tuple[int, ...]] = self.sid_to_core_tuple_hash
        sid_to_item1_set_hash: List[Item1Set] = self.sid_to_item1_set_hash
        item1_set_relation: List[Tuple[int, int, int]] = self.item1_set_relation

        # 根据入口项的 LR(0) 项构造 LR(1) 项
        init_i1_id = self._create_item1(self.init_item0, self.grammar.end_terminal)
        init_core_tuple = (init_i1_id,)
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
            other_item1_set = self.bfs_closure_item1(core_tuple)
            # other_item1_set = set()
            # for i1_id in core_tuple:
            #     other_item1_set |= self.dfs_closure_item1(i1_id)

            # 构造项目集闭包并添加到结果集中
            item1_set = Item1Set.create(
                sid=sid1,
                core_list=core_tuple,
                other_item1_set=other_item1_set
            )
            sid_to_item1_set_hash.append(item1_set)

            # 根据后继项目符号进行分组，计算出每个后继项目集闭包的核心项目元组
            successor_group = collections.defaultdict(list)
            for i1_id in item1_set.core_tuple:
                item1 = self.i1_id_to_item1_hash[i1_id]
                successor_symbol = item1.item0.successor_symbol
                if successor_symbol is not None:
                    successor_group[successor_symbol].append(item1.successor_item)
            for i1_id in item1_set.other_item1_set:
                item1 = self.i1_id_to_item1_hash[i1_id]
                successor_symbol = item1.item0.successor_symbol
                if successor_symbol is not None:
                    successor_group[successor_symbol].append(item1.successor_item)

            # 计算后继项目集的核心项目元组（排序以保证顺序稳定）
            successor_core_tuple_hash = {}
            for successor_symbol, sub_item1_set in successor_group.items():
                successor_core_tuple: Tuple[int, ...] = tuple(sorted(set(sub_item1_set)))
                if successor_core_tuple not in core_tuple_to_sid_hash:
                    successor_sid1 = len(core_tuple_to_sid_hash)
                    core_tuple_to_sid_hash[successor_core_tuple] = successor_sid1
                    sid_to_core_tuple_hash.append(successor_core_tuple)
                else:
                    successor_sid1 = core_tuple_to_sid_hash[successor_core_tuple]
                successor_core_tuple_hash[successor_symbol] = successor_sid1

            # 记录项目集闭包之间的关联关系
            for successor_symbol, successor_sid1 in successor_core_tuple_hash.items():
                item1_set_relation.append((sid1, successor_symbol, successor_sid1))

            # 将后继项目集闭包的核心项目元组添加到队列
            for successor_sid1 in successor_core_tuple_hash.values():
                if successor_sid1 not in visited:
                    queue.append(successor_sid1)
                    visited.add(successor_sid1)

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
        core_tuple : Tuple[Item1]
            项目集闭包的核心项目（最高层级项目）

        Returns
        -------
        List[Item1]
            项目集闭包中包含的项目列表
        """
        # 初始化项目集闭包中包含的其他项目列表
        item_set: Set[int] = set()

        # 初始化广度优先搜索的第 1 批节点
        visited_symbol_set = set()
        queue = collections.deque()
        for i1_id in core_tuple:
            item1 = self.i1_id_to_item1_hash[i1_id]
            after_handle = item1.item0.after_handle

            # 如果核心项是规约项目，则不存在等价项目组，跳过该项目即可
            if not after_handle:
                continue

            # 将句柄之后的符号列表 + 展望符添加到队列中
            visited_symbol_set.add((after_handle, item1.lookahead))
            queue.append((after_handle, item1.lookahead))

        # 广度优先搜索所有的等价项目组
        while queue:
            after_handle, lookahead = queue.popleft()

            # 计算单层的等价 LR(1) 项目
            sub_item_set = self.compute_single_level_lr1_closure(
                after_handle=after_handle,
                lookahead=lookahead
            )

            # 将当前项目组匹配的等价项目组添加到所有等价项目组中
            diff_set = sub_item_set - item_set
            item_set |= diff_set

            # 将等价项目组中需要继续寻找等价项目的添加到队列
            for i1_id in diff_set:
                sub_item1 = self.i1_id_to_item1_hash[i1_id]
                after_handle = sub_item1.item0.after_handle
                if not after_handle:
                    continue  # 跳过匹配 %empty 的项目

                lookahead = sub_item1.lookahead
                if (after_handle, lookahead) not in visited_symbol_set:
                    visited_symbol_set.add((after_handle, lookahead))
                    queue.append((after_handle, lookahead))

        return item_set

    @lru_cache(maxsize=None)
    def dfs_closure_item1(self, i1_id: int) -> Set[int]:
        """广度优先搜索，记忆化搜索，根据项目集核心项目元组（core_tuple）生成项目集闭包中包含的其他项目列表（item_list）"""
        item1 = self.i1_id_to_item1_hash[i1_id]
        print("DFS:", i1_id, item1)
        after_handle = item1.item0.after_handle
        if not after_handle:
            return set()

        self._dfs_visited.add(i1_id)

        # 计算单层的等价 LR(1) 项目
        item1_set = self.compute_single_level_lr1_closure(
            after_handle=after_handle,
            lookahead=item1.lookahead
        )
        for sub_i1_id in list(item1_set):
            if sub_i1_id not in self._dfs_visited:
                item1_set |= self.dfs_closure_item1(sub_i1_id)

        self._dfs_visited.remove(i1_id)

        return item1_set

    @lru_cache(maxsize=None)
    def compute_single_level_lr1_closure(self, after_handle: Tuple[int, ...], lookahead: int) -> Set[int]:
        """计算 item1 单层的等价 LR(1) 项目的 ID 的集合

        计算单层的等价 LR(1) 项目，即只将非终结符替换为等价的终结符或非终结符，但不会计算替换后的终结符的等价 LR(1) 项目。

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
        n_terminal = self.grammar.n_terminal  # 【性能设计】提前获取需频繁使用的 grammar 中的常量，以减少调用次数
        len_after_handle = len(after_handle)  # 【性能设计】提前计算需要频繁使用的常量

        # 如果开头符号是终结符，则不存在等价项目
        # 【性能】通过 first_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
        first_symbol = after_handle[0]
        if first_symbol < n_terminal:
            return set()

        sub_item_set: Set[int] = set()  # 当前项目组之后的所有可能的 lookahead

        # 添加生成 symbol 非终结符对应的所有项目，并将这些项目也加入继续寻找等价项目组的队列中
        for item0 in self.symbol_to_start_item_list_hash[first_symbol]:
            # 先从当前句柄后第 1 个元素向后继续遍历，添加自生型后继
            i = 1
            is_stop = False  # 是否已经找到不能匹配 %empty 的非终结符或终结符
            while i < len_after_handle:  # 向后逐个遍历符号，寻找展望符
                next_symbol = after_handle[i]

                # 如果遍历到的符号是终结符，则将该终结符添加为展望符，则标记 is_stop 并结束遍历
                # 【性能】通过 next_symbol < n_terminal 判断 next_symbol 是否为终结符，以节省对 grammar.is_terminal 方法的调用
                if next_symbol < n_terminal:
                    sub_item_set.add(self._create_item1(item0, next_symbol))  # 自生后继符
                    is_stop = True
                    break

                # 如果遍历到的符号是非终结符，则遍历该非终结符的所有可能的开头终结符添加为展望符
                for start_terminal in self.nonterminal_all_start_terminal[next_symbol]:
                    sub_item_set.add(self._create_item1(item0, start_terminal))  # 自生后继符

                # 如果遍历到的非终结符不能匹配 %emtpy，则标记 is_stop 并结束遍历
                if not self.grammar.is_maybe_empty(next_symbol):
                    is_stop = True
                    break

                i += 1

            # 如果没有遍历到不能匹配 %empty 的非终结符或终结符，则添加继承型后继
            if is_stop is False:
                sub_item_set.add(self._create_item1(item0, lookahead))  # 继承后继符

        return sub_item_set

    def cal_concentric_hash(self) -> Dict[Tuple[int, ...], List[Item1Set]]:
        """计算 LR(1) 的项目集核心，并根据项目集的核心（仅包含规约符、符号列表和句柄的核心项目元组）进行聚合

        Returns
        -------
        Dict[Tuple[int, ...], List[Item1Set]]
            根据项目集核心聚合后的项目集
        """
        # 【性能设计】初始化方法中频繁使用的类属性，以避免重复获取类属性
        sid_to_core_tuple_hash: List[Tuple[int, ...]] = self.sid_to_core_tuple_hash
        sid_to_item1_set_hash: List[Item1Set] = self.sid_to_item1_set_hash

        concentric_hash = collections.defaultdict(list)
        for sid in self.sid_set:
            core_tuple = sid_to_core_tuple_hash[sid]
            item1_set = sid_to_item1_set_hash[sid]
            centric_tuple = tuple(sorted(list(set(self.i1_id_to_item1_hash[i1_id].item0.i0_id for i1_id in core_tuple))))
            # 根据项目集核心进行聚合
            concentric_hash[centric_tuple].append(item1_set)
        return concentric_hash

    def merge_same_concentric_item1_set(self) -> None:
        # pylint: disable=R0914
        """合并 LR(1) 项目集核心相同的 LR(1) 项目集（原地更新）"""

        # 【性能设计】初始化方法中频繁使用的类属性，以避免重复获取类属性
        core_tuple_to_sid_hash: Dict[Tuple[int, ...], int] = self.core_tuple_to_sid_hash
        sid_to_core_tuple_hash: List[Tuple[int, ...]] = self.sid_to_core_tuple_hash
        sid_to_item1_set_hash: List[Item1Set] = self.sid_to_item1_set_hash

        for _, item1_set_list in self.concentric_hash.items():
            if len(item1_set_list) == 1:
                continue  # 如果没有项目集核心相同的多个项目集，则不需要合并

            # 构造新的项目集
            new_core_item_set: Set[int] = set()  # 新项目集的核心项目
            new_other_item_set: Set[int] = set()  # 新项目集的其他等价项目
            for item1_set in item1_set_list:
                new_core_item_set |= set(item1_set.core_tuple)
                new_other_item_set |= item1_set.other_item1_set

            # 通过排序逻辑以保证结果状态是稳定的
            new_core_item_list = list(new_core_item_set)
            new_core_item_list.sort(key=lambda x: self.i1_id_to_item1_hash[x])
            new_core_tuple = tuple(new_core_item_list)

            if new_core_tuple not in self.core_tuple_to_sid_hash:
                new_sid1 = len(self.core_tuple_to_sid_hash)
                core_tuple_to_sid_hash[new_core_tuple] = new_sid1
                sid_to_core_tuple_hash.append(new_core_tuple)
            else:
                new_sid1 = core_tuple_to_sid_hash[new_core_tuple]

            new_item1_set = Item1Set.create(
                sid=new_sid1,
                core_list=new_core_tuple,
                other_item1_set=new_other_item_set
            )
            sid_to_item1_set_hash.append(new_item1_set)

            # 记录旧 core_tuple 到新 core_tuple 的映射
            for item1_set in item1_set_list:
                self.core_tuple_hash[item1_set.sid] = new_sid1

            # 整理记录有效 SID1 的集合
            for item1_set in item1_set_list:
                if item1_set.sid in self.sid_set:
                    self.sid_set.remove(item1_set.sid)
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
        for s1_id in self.sid_set:
            status_id = self.sid_to_status_hash[s1_id]  # 计算 LR(1) 项目集的 S1_ID 对应的状态 ID
            item1_set = self.sid_to_item1_set_hash[s1_id]  # 计算 LR(1) 项目集的 S1_ID 对应的状态 ID

            # 根据项目集闭包的后继项目，填充 ACTION 表和 GOTO 表
            for successor_symbol, successor_s1_id in self.s1_id_relation[s1_id].items():
                next_status_id = self.sid_to_status_hash[successor_s1_id]
                if self.grammar.is_terminal(successor_symbol):
                    # 后继项目为终结符，记录需要填充到 ACTION 表的 Shift 行为
                    position_shift_hash[(status_id, successor_symbol)] = ActionShift(status=next_status_id)
                else:
                    # 后继项目为非终结符，填充 GOTO 表
                    table[status_id][successor_symbol] = ActionGoto(status=next_status_id)

            # 遍历不包含后继项目的项目，记录需要填充到 ACTION 表的 Reduce 行为
            for i1_id in item1_set.core_tuple:
                sub_item1 = self.i1_id_to_item1_hash[i1_id]
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
            for i1_id in item1_set.other_item1_set:
                sub_item1 = self.i1_id_to_item1_hash[i1_id]
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
